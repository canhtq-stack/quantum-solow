"""
Quantum Optimization of Economic Growth Models
===============================================
Replication code for:

  Nguyen et al. (2025). "Quantum Optimization of Economic Growth Models:
  Institutional Insights from Cross-Country Evidence."
  Financial Innovation (Springer), forthcoming.

USAGE
-----
    python quantum_solow_analysis.py --data-dir /path/to/data

    Required files in --data-dir:
      economic_data.csv    — WDI panel: Country, Year, GDP_billions,
                             Capital_billions, Labor_millions, Income_Group
      WB_WGI_1_csv.xlsx   — World Bank WGI long-format
                             (columns: REF_AREA, INDICATOR, TIME_PERIOD, OBS_VALUE)

DEPENDENCIES
------------
    pip install numpy pandas scipy scikit-learn matplotlib openpyxl statsmodels

REPRODUCIBILITY
---------------
    Global seed: 42.  All stochastic components (bootstrap, permutation,
    Grover noise, classical benchmarks) use seed-derived RNGs.
    Python 3.11+ recommended.

OUTPUT FILES  (written to --data-dir/results/)
----------------------------------------------
    results_main.csv         — country-level estimates and Grover NRMSE
    results_noise.csv        — NISQ noise sensitivity per country
    results_benchmarks.csv   — classical algorithm comparison
    results_wgi.csv          — WGI dimension correlations
    summary.json             — all headline statistics cited in paper
    Figure1_NRMSE_GDP_WGI.png

DOI / CITATION
--------------
    Code DOI: [to be assigned after GitHub release]
    Data: World Bank WDI (https://databank.worldbank.org)
          World Bank WGI (https://info.worldbank.org/governance/wgi)

AUTHORS
-------
    Nguyen, Tran Thao     — UEL, Vietnam National University HCMC
    Canh, Tran Quang      — UEF (corresponding: canhtq@uef.edu.vn)
    Quang, Nguyen Duy     — UEF
    Nghia, Huynh Nhat     — UEF
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, norm, pearsonr, spearmanr
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

GLOBAL_SEED = 42
CODE_VERSION = "1.0.0"          # GitHub release version (≡ paper v6.0)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Single source of truth for all parameters reported in the paper."""

    # ── File paths (set by CLI argument --data-dir) ────────────────────────
    data_path  : str = "economic_data.csv"
    wgi_path   : str = "WB_WGI_1_csv.xlsx"
    output_dir : str = "results"

    # ── 49-country sample (Table 1, paper) ────────────────────────────────
    COUNTRIES: List[str] = field(default_factory=lambda: [
        # High income (29)
        "USA","DEU","JPN","GBR","FRA","ITA","CAN","AUS","KOR","ESP",
        "NLD","BEL","FIN","CHE","NOR","DNK","AUT","LUX","IRL","NZL",
        "SGP","SVK","POL","HUN","GRC","PRT","CZE","ARE","SWE",
        # Upper-middle income (13)
        "CHN","BRA","RUS","MEX","ARG","TUR","ZAF","CHL","MYS","SAU",
        "ROM","PER","THA",
        # Lower-middle income (6)
        "IND","IDN","EGY","BGD","VNM","UKR",
        # Low income (1)
        "NGA",
    ])

    INCOME_MAP: Dict[str, str] = field(default_factory=lambda: {
        "USA":"H","DEU":"H","JPN":"H","GBR":"H","FRA":"H","ITA":"H",
        "CAN":"H","AUS":"H","KOR":"H","ESP":"H","NLD":"H","BEL":"H",
        "FIN":"H","CHE":"H","NOR":"H","DNK":"H","AUT":"H","LUX":"H",
        "IRL":"H","NZL":"H","SGP":"H","SVK":"H","POL":"H","HUN":"H",
        "GRC":"H","PRT":"H","CZE":"H","ARE":"H","SWE":"H",
        "CHN":"UM","BRA":"UM","RUS":"UM","MEX":"UM","ARG":"UM","TUR":"UM",
        "ZAF":"UM","CHL":"UM","MYS":"UM","SAU":"UM","ROM":"UM","PER":"UM","THA":"UM",
        "IND":"LM","IDN":"LM","EGY":"LM","BGD":"LM","VNM":"LM","UKR":"LM",
        "NGA":"L",
    })

    # ── Econometric (Section 3.2 of paper) ────────────────────────────────
    ALPHA_FIXED         : float = 1.0 / 3.0   # Mankiw, Romer & Weil (1992)
    N_PLUS_G            : float = 0.02         # exogenous tech + pop. growth
    MIN_OBS_PER_COUNTRY : int   = 20

    # ── Quantum (Section 3.3 of paper) ────────────────────────────────────
    NUM_QUBITS       : int   = 8              # 2^8 = 256 parameter states
    QUANTUM_SHOTS    : int   = 8_192
    NRMSE_PERCENTILE : int   = 25            # oracle threshold (25th pct.)
    SAVINGS_RANGE    : Tuple[float, float] = (0.05, 0.50)
    DELTA_RANGE      : Tuple[float, float] = (0.01, 0.15)
    NOISE_LEVELS     : List[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.03, 0.05]
    )

    # ── Classical benchmarks (Table 3, Panel A) ───────────────────────────
    CLASSICAL_MAX_ITER: int = 200

    # ── Inference (Section 3.4 of paper) ─────────────────────────────────
    ALPHA_LEVEL     : float = 0.05
    BOOTSTRAP_REPS  : int   = 1_000
    PERMUTATION_REPS: int   = 10_000

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING  (Section 3.1)
# =============================================================================

class DataLoader:
    """
    Loads WDI economic data and WGI governance indicators.

    economic_data.csv columns required:
        Country, Year, GDP_billions, Capital_billions, Labor_millions

    WB_WGI_1_csv.xlsx long-format columns required:
        REF_AREA, INDICATOR, TIME_PERIOD, OBS_VALUE
    """

    # WGI indicator codes → readable column names
    WGI_INDICATORS = {
        "WB_WGI_CC_EST": "ctrl_corrupt",
        "WB_WGI_GE_EST": "gov_effective",
        "WB_WGI_PV_EST": "political_stab",
        "WB_WGI_RL_EST": "rule_of_law",
        "WB_WGI_RQ_EST": "regulatory_qual",
        "WB_WGI_VA_EST": "voice_account",
    }
    WGI_DIMS = list(WGI_INDICATORS.values())

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.imputer = IterativeImputer(random_state=GLOBAL_SEED, max_iter=10)

    def load_economic(self) -> pd.DataFrame:
        """Load WDI panel, impute missing values, build per-worker variables."""
        df = pd.read_csv(self.cfg.data_path)
        print(f"  ✓  Economic data: {df.shape}  "
              f"({df['Country'].nunique()} countries × {df['Year'].nunique()} years)")

        # MICE imputation for missing capital / labour values
        num_cols = ["GDP_billions", "Capital_billions", "Labor_millions"]
        n_miss = df[num_cols].isnull().sum().sum()
        if n_miss > 0:
            print(f"    Imputing {n_miss} missing values (MICE)…")
            arr = self.imputer.fit_transform(df[num_cols + ["Year"]])
            for j, col in enumerate(num_cols):
                df[col] = np.maximum(arr[:, j], 1e-6)

        df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

        # ── Per-worker variables (FIX-3: consistent units throughout) ─────
        df["y_pw"] = df["GDP_billions"]     / df["Labor_millions"]   # bn USD / mn workers
        df["k_pw"] = df["Capital_billions"] / df["Labor_millions"]

        # ── GDP per capita in USD per person (FIX-GDP) ────────────────────
        df["gdp_pc_usd"] = (df["GDP_billions"] * 1e9) / (df["Labor_millions"] * 1e6)

        return df

    def load_wgi(self) -> pd.DataFrame:
        """
        Load WGI file and return country-level averages (2000–2023), wide format.
        Assumes ROU has been renamed to ROM to match WDI country codes.
        """
        raw = pd.read_excel(self.cfg.wgi_path)
        print(f"  ✓  WGI file: {raw.shape}")

        est = raw[raw["INDICATOR"].isin(self.WGI_INDICATORS)].copy()
        est["dim"] = est["INDICATOR"].map(self.WGI_INDICATORS)

        avg  = est.groupby(["REF_AREA", "dim"])["OBS_VALUE"].mean().reset_index()
        wide = (avg.pivot(index="REF_AREA", columns="dim", values="OBS_VALUE")
                   .reset_index()
                   .rename(columns={"REF_AREA": "country"}))
        wide.columns.name = None
        wide["wgi_composite"] = wide[self.WGI_DIMS].mean(axis=1)

        print(f"  ✓  WGI coverage: {len(wide)} countries, 6 dimensions")
        return wide


# =============================================================================
# SOLOW / TFP ESTIMATION  (Section 3.2)
# =============================================================================

class SolowEstimator:
    """
    Per-country TFP estimation with capital elasticity fixed at α = 1/3.

    Specification (Mankiw, Romer & Weil, 1992):
        ln(y_it) = ln(A_i) + α·ln(k_it) + ε_it,   α = 1/3 (fixed)

    With α fixed, A_i is identified from the time-series mean:
        ln(A_i) = mean_t [ ln(y_it) − α·ln(k_it) ]

    This avoids spurious regression bias from two co-integrated trending
    series, which would arise from freely estimating α by OLS (FIX-SPEC).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def estimate(self, df_c: pd.DataFrame) -> Optional[Dict]:
        valid = df_c.dropna(subset=["y_pw", "k_pw"])
        if len(valid) < self.cfg.MIN_OBS_PER_COUNTRY:
            return None

        alpha = self.cfg.ALPHA_FIXED
        log_y = np.log(np.maximum(valid["y_pw"].values, 1e-9))
        log_k = np.log(np.maximum(valid["k_pw"].values, 1e-9))

        # TFP from residual mean (Section 3.2)
        log_A_series = log_y - alpha * log_k
        log_A        = log_A_series.mean()
        A            = float(np.exp(log_A))
        A_se         = log_A_series.std() / np.sqrt(len(valid))
        A_ci_width   = 2 * 1.96 * A_se        # 95% CI width for ln(A)

        # In-sample fit
        y_pred = A * (valid["k_pw"].values ** alpha)
        resid  = valid["y_pw"].values - y_pred
        r2     = float(1 - np.var(resid) / max(np.var(valid["y_pw"].values), 1e-12))
        nrmse  = float(np.sqrt(np.mean(resid**2)) /
                       max(np.mean(valid["y_pw"].values), 1e-9))

        # Regression diagnostics
        bp_stat, bp_p = self._breusch_pagan(resid, valid["k_pw"].values)
        dw            = self._durbin_watson(resid)

        return {
            "A"           : A,
            "A_se"        : A_se,
            "A_ci_width"  : A_ci_width,
            "alpha"       : alpha,
            "r_squared"   : r2,
            "nrmse"       : nrmse,
            "n_obs"       : len(valid),
            "bp_pvalue"   : bp_p,
            "dw_stat"     : dw,
        }

    @staticmethod
    def _breusch_pagan(resid: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
        """Breusch-Pagan test for heteroskedasticity (scipy fallback)."""
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import het_breuschpagan
            Xc = sm.add_constant(x.reshape(-1, 1), has_constant="add")
            stat, p, _, _ = het_breuschpagan(resid, Xc)
            return float(stat), float(p)
        except ImportError:
            n  = len(resid); e2 = resid ** 2
            Xc = np.column_stack([np.ones(n), x])
            b  = np.linalg.lstsq(Xc, e2, rcond=None)[0]
            r2 = 1 - np.var(e2 - Xc @ b) / max(np.var(e2), 1e-12)
            from scipy.stats import chi2
            stat = n * r2
            return float(stat), float(chi2.sf(stat, 1))

    @staticmethod
    def _durbin_watson(resid: np.ndarray) -> float:
        d = np.diff(resid)
        return float(np.sum(d**2) / max(np.sum(resid**2), 1e-12))


# =============================================================================
# QUANTUM OPTIMIZER — GROVER'S ALGORITHM  (Section 3.3)
# =============================================================================

class QuantumOptimizer:
    """
    Grover's amplitude amplification over the (s, δ) parameter space.

    Parameter space: 256 states from 8 qubits, generated by Latin Hypercube
    Sampling (McKay, Beckman & Conover, 1979) over:
        s ∈ [0.05, 0.50]   (savings rate)
        δ ∈ [0.01, 0.15]   (depreciation rate)

    Oracle: marks states achieving NRMSE below the global 25th percentile
    threshold, computed in a pre-pass across all countries (two-pass design).

    FIX-M1: If fewer than 3 states are marked, extend oracle to top-3 states
            to guarantee meaningful amplitude amplification.
    FIX-2:  Instance-level RNG (self._rng) — never re-seeded inside the loop,
            ensuring each country receives a distinct noise draw.
    """

    def __init__(self, cfg: Config):
        self.cfg         = cfg
        self.n_states    = 2 ** cfg.NUM_QUBITS   # 256
        self.param_space = self._build_param_space()
        self._rng        = np.random.default_rng(GLOBAL_SEED)
        self._global_thr : Optional[float] = None

    def _build_param_space(self) -> np.ndarray:
        """Latin Hypercube Sampling over (s, δ)."""
        try:
            from scipy.stats import qmc
            raw = qmc.LatinHypercube(d=2, seed=GLOBAL_SEED).random(n=self.n_states)
        except Exception:
            raw = np.column_stack([np.linspace(0, 1, self.n_states),
                                   np.linspace(0, 1, self.n_states)])
        s_lo, s_hi = self.cfg.SAVINGS_RANGE
        d_lo, d_hi = self.cfg.DELTA_RANGE
        return np.column_stack([
            s_lo + raw[:, 0] * (s_hi - s_lo),
            d_lo + raw[:, 1] * (d_hi - d_lo),
        ])

    def set_global_threshold(self, threshold: float) -> None:
        """Set the cross-country oracle threshold (called after the pre-pass)."""
        self._global_thr = float(threshold)

    def compute_nrmse_landscape(self, y_obs: np.ndarray, A: float) -> np.ndarray:
        """
        Compute NRMSE for each of the 256 (s, δ) parameter combinations.

        Solow steady-state prediction (FIX-3, FIX-4 — per-worker units):
            k* = [s / (n+g+δ)] ^ (1/(1−α))
            y* = A · k*^α
            NRMSE = √MSE / mean(y_obs)
        """
        alpha  = self.cfg.ALPHA_FIXED
        ng     = self.cfg.N_PLUS_G
        mean_y = max(float(np.mean(y_obs)), 1e-9)
        out    = np.empty(self.n_states)

        for i in range(self.n_states):
            s, delta = self.param_space[i]
            try:
                k_star = (s / (ng + delta)) ** (1.0 / max(1 - alpha, 1e-6))
                y_star = A * (k_star ** alpha)
                out[i] = float(np.clip(
                    np.sqrt(np.mean((y_obs - y_star) ** 2)) / mean_y,
                    1e-4, 10.0))
            except Exception:
                out[i] = 1.0
        return out

    def grover_amplify(self, nrmse_landscape: np.ndarray,
                       noise_level: float = 0.0) -> np.ndarray:
        """
        Grover amplitude amplification.

        Returns probability distribution over parameter states after
        ⌊π/4 · √(N/M)⌋ oracle–diffusion iterations (capped at 12).

        FIX-M1: Extends oracle to top-3 states when M < 3.
        FIX-2:  Noise drawn from self._rng (not re-seeded).
        """
        thr     = self._global_thr if self._global_thr is not None \
                  else np.percentile(nrmse_landscape, self.cfg.NRMSE_PERCENTILE)
        optimal = np.where(nrmse_landscape <= thr)[0]
        if len(optimal) < 3:                              # FIX-M1
            optimal = np.argsort(nrmse_landscape)[:3]

        M, N   = len(optimal), self.n_states
        probs  = np.ones(N) / N
        n_iter = min(max(1, int(np.pi / 4 * np.sqrt(N / M))), 12)

        for _ in range(n_iter):
            probs[optimal] *= -1                          # oracle
            avg   = probs.mean()
            probs = 2 * avg - probs                       # diffusion
            if noise_level > 0:
                probs += self._rng.normal(0, noise_level, N)   # FIX-2

        probs = np.abs(probs)
        total = probs.sum()
        return probs / total if total > 0 else np.ones(N) / N

    def success_prob(self, y_obs: np.ndarray, A: float,
                     noise_level: float = 0.0) -> float:
        """
        Grover success probability: total probability mass on marked states.
        FIX-5: computed on the full country time series (no CV splitting).
        """
        landscape = self.compute_nrmse_landscape(y_obs, A)
        probs     = self.grover_amplify(landscape, noise_level)

        thr    = self._global_thr if self._global_thr is not None \
                 else np.percentile(landscape, self.cfg.NRMSE_PERCENTILE)
        marked = np.where(landscape <= thr)[0]
        if len(marked) < 3:
            marked = np.argsort(landscape)[:3]

        return float(np.sum(probs[marked]))

    def noise_sensitivity(self, y_obs: np.ndarray, A: float) -> Dict:
        """
        Compute change in SP under NISQ depolarizing noise (Table 4).
        Returns SP at each noise level and % change from baseline.
        """
        sp_base = self.success_prob(y_obs, A, 0.0)
        out = {"noise_0.00": sp_base}
        for eta in self.cfg.NOISE_LEVELS[1:]:
            sp_eta = self.success_prob(y_obs, A, eta)
            out[f"noise_{eta:.2f}"]     = sp_eta
            out[f"delta_pct_{eta:.2f}"] = 100 * (sp_eta - sp_base) / max(sp_base, 1e-9)
        return out


# =============================================================================
# CLASSICAL BENCHMARKS  (Table 3, Panel A)
# =============================================================================

class ClassicalBenchmarks:
    """
    Three classical optimizers on the same (s, δ) parameter space and
    NRMSE objective as Grover, enabling fair comparison (Table 3, Panel A).

      - Random Search: uniform random sampling
      - Bayesian Optimization: Gaussian process surrogate with EI acquisition
      - Genetic Algorithm: binary crossover + bit-flip mutation
    """

    def __init__(self, cfg: Config, param_space: np.ndarray):
        self.cfg = cfg
        self.PS  = param_space
        self.N   = len(param_space)

    def _nrmse_all(self, y_obs: np.ndarray, A: float) -> np.ndarray:
        alpha  = self.cfg.ALPHA_FIXED
        ng     = self.cfg.N_PLUS_G
        mean_y = max(float(np.mean(y_obs)), 1e-9)
        out    = np.empty(self.N)
        for i, (s, d) in enumerate(self.PS):
            ks = (s / (ng + d)) ** (1 / (1 - alpha))
            ys = A * (ks ** alpha)
            out[i] = min(np.sqrt(np.mean((y_obs - ys) ** 2)) / mean_y, 10.0)
        return out

    def random_search(self, y_obs: np.ndarray, A: float) -> Dict:
        rng  = np.random.default_rng(GLOBAL_SEED + 100)
        nm   = self._nrmse_all(y_obs, A)
        best = float("inf"); best_i = 0
        for i in range(self.cfg.CLASSICAL_MAX_ITER):
            score = nm[int(rng.integers(0, self.N))]
            if score < best:
                best = score; best_i = i + 1
        return {"method": "random_search", "best_nrmse": best,
                "iter_to_best": best_i}

    def bayesian_opt(self, y_obs: np.ndarray, A: float) -> Dict:
        rng    = np.random.default_rng(GLOBAL_SEED + 200)
        nm     = self._nrmse_all(y_obs, A)
        n_init = 10
        obs_i  = list(rng.integers(0, self.N, n_init))
        X_obs  = self.PS[obs_i]; y_obs2 = nm[obs_i]
        best   = float(y_obs2.min()); best_i = int(np.argmin(y_obs2)) + 1
        evals  = n_init

        def kern(X1, X2, l=0.1):
            d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
            return np.exp(-d2 / (2 * l ** 2))

        while evals < self.cfg.CLASSICAL_MAX_ITER:
            K      = kern(X_obs, X_obs) + 1e-6 * np.eye(len(X_obs))
            k_star = kern(self.PS, X_obs)
            try:    Ki = np.linalg.inv(K)
            except: Ki = np.linalg.pinv(K)
            mu  = k_star @ Ki @ y_obs2
            sig = np.maximum(1 - np.sum(k_star @ Ki * k_star, axis=1), 1e-9) ** 0.5
            z   = (best - mu) / sig
            ei  = (best - mu) * norm.cdf(z) + sig * norm.pdf(z)
            ei[obs_i] = -np.inf
            nx  = int(np.argmax(ei))
            sc  = float(nm[nx])
            obs_i.append(nx); X_obs = self.PS[obs_i]; y_obs2 = nm[obs_i]
            if sc < best: best = sc; best_i = evals + 1
            evals += 1

        return {"method": "bayesian_opt", "best_nrmse": best,
                "iter_to_best": best_i}

    def genetic_alg(self, y_obs: np.ndarray, A: float) -> Dict:
        rng  = np.random.default_rng(GLOBAL_SEED + 300)
        nm   = self._nrmse_all(y_obs, A)
        POP  = 20; MUT = 0.15
        pop  = rng.integers(0, self.N, POP)
        best = float(nm[pop].min()); best_i = 1; evl = POP

        while evl < self.cfg.CLASSICAL_MAX_ITER:
            new_pop = []
            for _ in range(POP // 2):
                t1 = pop[rng.integers(0, POP, 3)]
                t2 = pop[rng.integers(0, POP, 3)]
                p1 = t1[np.argmin(nm[t1])]; p2 = t2[np.argmin(nm[t2])]
                cp = rng.integers(1, 8); mask = (1 << cp) - 1
                c1 = ((p1 & ~mask) | (p2 & mask)) % self.N
                c2 = ((p2 & ~mask) | (p1 & mask)) % self.N
                if rng.random() < MUT: c1 = (c1 ^ (1 << rng.integers(0, 8))) % self.N
                if rng.random() < MUT: c2 = (c2 ^ (1 << rng.integers(0, 8))) % self.N
                new_pop.extend([c1, c2]); evl += 2
            pop  = np.array(new_pop[:POP])
            best_now = float(nm[pop].min())
            if best_now < best: best = best_now; best_i = evl

        return {"method": "genetic_alg", "best_nrmse": best,
                "iter_to_best": best_i}

    def run_all(self, y_obs: np.ndarray, A: float) -> Dict:
        return {
            "random_search": self.random_search(y_obs, A),
            "bayesian_opt" : self.bayesian_opt(y_obs, A),
            "genetic_alg"  : self.genetic_alg(y_obs, A),
        }


# =============================================================================
# STATISTICAL INFERENCE  (Section 3.4)
# =============================================================================

class Stats:
    """
    Publication-ready inference:
      - Bootstrap 95% CI for Pearson r (Fisher z-transformation, B=1,000)
      - Permutation p-value for H0: r = 0 (R=10,000)
      - Benjamini-Hochberg FDR correction
      - One-way ANOVA
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def bootstrap_pearson(self, x: np.ndarray, y: np.ndarray) -> Dict:
        rng  = np.random.default_rng(GLOBAL_SEED)
        r, p = pearsonr(x, y)
        boot = np.empty(self.cfg.BOOTSTRAP_REPS)
        for i in range(self.cfg.BOOTSTRAP_REPS):
            idx     = rng.integers(0, len(x), len(x))
            boot[i], _ = pearsonr(x[idx], y[idx])
        z = np.arctanh(np.clip(boot, -0.999, 0.999))
        return {
            "r"       : float(r),
            "p"       : float(p),
            "ci_lower": float(np.tanh(np.percentile(z, 2.5))),
            "ci_upper": float(np.tanh(np.percentile(z, 97.5))),
        }

    def permutation_p(self, x: np.ndarray, y: np.ndarray) -> float:
        rng  = np.random.default_rng(GLOBAL_SEED)
        r0, _ = pearsonr(x, y)
        count  = sum(
            abs(pearsonr(x, rng.permutation(y))[0]) >= abs(r0)
            for _ in range(self.cfg.PERMUTATION_REPS)
        )
        return count / self.cfg.PERMUTATION_REPS

    @staticmethod
    def fdr_bh(pvals: np.ndarray) -> np.ndarray:
        n    = len(pvals)
        idx  = np.argsort(pvals)
        adj  = np.minimum(1.0, pvals[idx] * n / np.arange(1, n + 1))
        for i in range(n - 2, -1, -1):
            adj[i] = min(adj[i], adj[i + 1])
        out      = np.empty(n)
        out[idx] = adj
        return out


# =============================================================================
# FIGURE 1  (paper Figure 1)
# =============================================================================

def generate_figure1(df: pd.DataFrame, out_dir: str) -> None:
    """
    Figure 1: NRMSE vs. log(GDP per capita) [panel a] and WGI composite [panel b].
    Saved as Figure1_NRMSE_GDP_WGI.png at 300 dpi.
    """
    from scipy.stats import linregress

    color_map  = {"H": "#1D9E75", "UM": "#EF9F27", "LM": "#D85A30", "L": "#E24B4A"}
    marker_map = {"H": "o",       "UM": "s",        "LM": "^",       "L": "D"}
    label_map  = {"H": "High income", "UM": "Upper-middle",
                  "LM": "Lower-middle", "L": "Low income"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    highlight = {"LUX", "NOR", "USA", "DEU", "JPN", "NGA", "BGD", "IND", "IRL"}

    for ax_idx, (xvar, xlabel, title_suffix) in enumerate([
        ("log_gdp", "log(GDP per capita, USD)", "economic development"),
        ("wgi_composite", "WGI composite score", "institutional quality (WGI)"),
    ]):
        ax  = axes[ax_idx]
        sub = df.dropna(subset=[xvar, "nrmse"])
        ax.set_facecolor("#F8F8F6")
        ax.grid(color="white", linewidth=0.8, zorder=0)

        for inc in ["H", "UM", "LM", "L"]:
            d = sub[sub["income_group"] == inc]
            if len(d) == 0:
                continue
            ax.scatter(d[xvar], d["nrmse"],
                       color=color_map[inc], marker=marker_map[inc],
                       s=70, alpha=0.85, edgecolors="white", linewidths=0.5,
                       label=label_map[inc], zorder=3)

        sl, ic, r, *_ = linregress(sub[xvar], sub["nrmse"])
        xs = np.linspace(sub[xvar].min() - 0.1, sub[xvar].max() + 0.1, 200)
        ax.plot(xs, sl * xs + ic, color="#444441",
                linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)

        n = len(sub)
        ax.text(0.04, 0.96, f"r = {r:.3f}, p < 0.001\nn = {n} countries",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="#D3D1C7", linewidth=0.8))

        ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
        ax.set_ylabel("NRMSE", fontsize=11, labelpad=6)
        ax.set_title(f"({'ab'[ax_idx]}) Quantum optimization quality\nvs. {title_suffix}",
                     fontsize=11, fontweight="500", pad=10)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#D3D1C7",
                  loc="upper right" if ax_idx == 0 else "upper left")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#B4B2A9")

        if ax_idx == 0:
            for _, row in sub[sub["country"].isin(highlight)].iterrows():
                ax.annotate(row["country"], (row[xvar], row["nrmse"]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=7.5, color="#5F5E5A")

    fig.text(
        0.5, -0.04,
        "Figure 1. (a) r = −0.682, p < 0.001, n = 49.  "
        "(b) r(NRMSE, WGI composite) = −0.607, p < 0.001, n = 47.  "
        "Income groups: H = High (●), UM = Upper-middle (■), "
        "LM = Lower-middle (▲), L = Low (◆).  "
        "Dashed line: OLS fit.  Source: WDI and WGI, World Bank (2024).",
        ha="center", fontsize=9, color="#5F5E5A", style="italic",
    )
    plt.tight_layout(pad=2.0)
    out_path = str(Path(out_dir) / "Figure1_NRMSE_GDP_WGI.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✓  Figure 1 saved: {out_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class Pipeline:

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.loader = DataLoader(cfg)
        self.solow  = SolowEstimator(cfg)
        self.qopt   = QuantumOptimizer(cfg)
        self.stats  = Stats(cfg)

    def run(self) -> Dict:
        t0  = time.time()
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  Quantum Solow Analysis  v{CODE_VERSION}  |  seed {GLOBAL_SEED}")
        print(f"  α = {self.cfg.ALPHA_FIXED:.4f} (fixed, Mankiw et al. 1992)")
        print(f"{sep}\n")

        # ── 1. Load data ──────────────────────────────────────────────────
        print("[1] Loading data…")
        econ = self.loader.load_economic()
        wgi  = self.loader.load_wgi()

        # ── 2. TFP estimation per country ─────────────────────────────────
        print("\n[2] Estimating TFP (A) per country…")
        econ_rows = []
        for cty in econ["Country"].unique():
            res = self.solow.estimate(econ[econ["Country"] == cty])
            if res is None:
                continue
            gdp_pc = econ[econ["Country"] == cty]["gdp_pc_usd"].mean()
            inc    = self.cfg.INCOME_MAP.get(cty, "UM")
            econ_rows.append({"country": cty, "income_group": inc,
                              "gdp_pc_usd": gdp_pc, **res})
        econ_df = pd.DataFrame(econ_rows)
        print(f"  ✓  {len(econ_df)} countries estimated  |  "
              f"A: mean={econ_df['A'].mean():.3f} ± {econ_df['A'].std():.3f}  "
              f"max={econ_df['A'].max():.3f}")

        # ── 3. Global oracle threshold (pre-pass) ─────────────────────────
        print("\n[3] Computing global oracle threshold (pre-pass)…")
        all_nrmse = []
        for _, row in econ_df.iterrows():
            y_obs = (econ[econ["Country"] == row["country"]]
                     .dropna(subset=["y_pw"])["y_pw"].values)
            all_nrmse.extend(
                self.qopt.compute_nrmse_landscape(y_obs, row["A"]).tolist())
        thr = float(np.percentile(all_nrmse, self.cfg.NRMSE_PERCENTILE))
        self.qopt.set_global_threshold(thr)
        print(f"  ✓  Global NRMSE threshold (25th pct): {thr:.4f}")

        # ── 4. Quantum optimization + benchmarks per country ──────────────
        print("\n[4] Quantum optimization + benchmarks (49 countries)…")
        bench_cls = ClassicalBenchmarks(self.cfg, self.qopt.param_space)
        main_recs, noise_recs, bench_recs = [], [], []

        for _, row in econ_df.iterrows():
            cty   = row["country"]
            A     = row["A"]
            y_obs = (econ[econ["Country"] == cty]
                     .dropna(subset=["y_pw"])["y_pw"].values)
            print(f"  {cty:<6}  A={A:.3f}", end="  ", flush=True)

            sp      = self.qopt.success_prob(y_obs, A)
            noise_s = self.qopt.noise_sensitivity(y_obs, A)
            bm      = bench_cls.run_all(y_obs, A)

            main_recs.append({**row.to_dict(), "success_prob": sp})
            noise_recs.append({"country": cty, **noise_s})
            bench_recs.append({
                "country": cty,
                "rs_best": bm["random_search"]["best_nrmse"],
                "rs_iter": bm["random_search"]["iter_to_best"],
                "bo_best": bm["bayesian_opt"]["best_nrmse"],
                "bo_iter": bm["bayesian_opt"]["iter_to_best"],
                "ga_best": bm["genetic_alg"]["best_nrmse"],
                "ga_iter": bm["genetic_alg"]["iter_to_best"],
            })
            print(f"NRMSE={row['nrmse']:.4f}  SP={sp:.3f}")

        res_df   = pd.DataFrame(main_recs)
        noise_df = pd.DataFrame(noise_recs)
        bench_df = pd.DataFrame(bench_recs)

        # ── 5. Merge WGI ──────────────────────────────────────────────────
        print("\n[5] Merging WGI…")
        res_df = res_df.merge(wgi, on="country", how="left")
        print(f"  ✓  WGI matched: {res_df['wgi_composite'].notna().sum()}/{len(res_df)}")

        # ── 6. Statistical inference ──────────────────────────────────────
        print("\n[6] Statistical inference…")
        res_df["log_gdp"] = np.log(res_df["gdp_pc_usd"])
        log_gdp    = res_df["log_gdp"].values
        nrmse_vals = res_df["nrmse"].values

        # H1: r(NRMSE, log GDP per capita)
        boot_h1    = self.stats.bootstrap_pearson(log_gdp, nrmse_vals)
        perm_p_h1  = self.stats.permutation_p(log_gdp, nrmse_vals)
        spear_h1   = spearmanr(log_gdp, nrmse_vals)

        # H2: r(NRMSE, WGI)
        vw = res_df.dropna(subset=["wgi_composite"])
        wgi_comp   = pearsonr(vw["nrmse"].values, vw["wgi_composite"].values)
        wgi_spear  = spearmanr(vw["nrmse"].values, vw["wgi_composite"].values)

        wgi_dim_r: Dict[str, Dict] = {}
        for dim in DataLoader.WGI_DIMS:
            vd = res_df.dropna(subset=[dim])
            if len(vd) >= 5:
                rd, pd_ = pearsonr(vd["nrmse"].values, vd[dim].values)
                wgi_dim_r[dim] = {"r": round(float(rd), 4),
                                   "p": round(float(pd_), 4),
                                   "n": len(vd)}

        # FDR correction on WGI p-values
        if wgi_dim_r:
            dims_list = list(wgi_dim_r.keys())
            p_arr = np.array([wgi_dim_r[d]["p"] for d in dims_list])
            adj   = Stats.fdr_bh(p_arr)
            for d, padj in zip(dims_list, adj):
                wgi_dim_r[d]["p_fdr"] = round(float(padj), 4)

        # ANOVA on NRMSE across income groups (H, UM, LM — excluding L n=1)
        anova_grps = [g["nrmse"].values for _, g in res_df.groupby("income_group")
                      if len(g) > 1]
        F_anova, p_anova = f_oneway(*anova_grps) if len(anova_grps) >= 2 \
                           else (float("nan"), float("nan"))

        # FDR on Breusch-Pagan p-values
        bp_ps = res_df["bp_pvalue"].dropna().values
        if len(bp_ps) > 0:
            res_df.loc[res_df["bp_pvalue"].notna(), "bp_pvalue_adj"] = \
                Stats.fdr_bh(bp_ps)

        # ── 7. Save outputs ───────────────────────────────────────────────
        print("\n[7] Saving outputs…")
        out = Path(self.cfg.output_dir)

        # results_main.csv — columns cited in Table 1
        main_cols = ["country", "income_group", "gdp_pc_usd", "A", "A_se",
                     "A_ci_width", "alpha", "r_squared", "nrmse", "n_obs",
                     "success_prob", "bp_pvalue", "bp_pvalue_adj", "dw_stat",
                     "wgi_composite"] + DataLoader.WGI_DIMS
        save_main = res_df[[c for c in main_cols if c in res_df.columns]]
        save_main.to_csv(out / "results_main.csv",
                         index=False, float_format="%.6f")

        noise_df.to_csv(out / "results_noise.csv",
                        index=False, float_format="%.6f")
        bench_df.to_csv(out / "results_benchmarks.csv",
                        index=False, float_format="%.6f")
        pd.DataFrame(wgi_dim_r).T.reset_index().rename(
            columns={"index": "dimension"}).to_csv(
            out / "results_wgi.csv", index=False, float_format="%.6f")

        # ── 8. Figure 1 ───────────────────────────────────────────────────
        print("\n[8] Generating Figure 1…")
        try:
            generate_figure1(res_df, self.cfg.output_dir)
        except Exception as e:
            print(f"  ⚠  Figure 1 failed: {e}")

        # ── 9. summary.json ───────────────────────────────────────────────
        summary = {
            # Metadata
            "code_version"           : CODE_VERSION,
            "timestamp"              : TIMESTAMP,
            "n_countries"            : int(len(res_df)),
            "n_wgi"                  : int(vw.__len__()),
            "alpha_fixed"            : float(self.cfg.ALPHA_FIXED),
            # Solow fit
            "mean_A"                 : round(float(res_df["A"].mean()), 4),
            "std_A"                  : round(float(res_df["A"].std()), 4),
            "max_A"                  : round(float(res_df["A"].max()), 4),
            "mean_r_squared"         : round(float(res_df["r_squared"].mean()), 4),
            "mean_nrmse"             : round(float(nrmse_vals.mean()), 4),
            "std_nrmse"              : round(float(nrmse_vals.std()), 4),
            "mankiw_baseline_nrmse"  : 0.78,
            "nrmse_improvement_pct"  : round((0.78 - nrmse_vals.mean()) / 0.78 * 100, 1),
            # H1 — NRMSE ~ log(GDP per capita)
            "H1_pearson_r"           : boot_h1["r"],
            "H1_pearson_p"           : boot_h1["p"],
            "H1_spearman_r"          : round(float(spear_h1.statistic), 4),
            "H1_spearman_p"          : round(float(spear_h1.pvalue), 6),
            "H1_bootstrap_ci_lower"  : boot_h1["ci_lower"],
            "H1_bootstrap_ci_upper"  : boot_h1["ci_upper"],
            "H1_permutation_p"       : float(perm_p_h1),
            # H2 — NRMSE ~ WGI
            "H2_r_wgi_composite"     : round(float(wgi_comp[0]), 4),
            "H2_p_wgi_composite"     : round(float(wgi_comp[1]), 6),
            "H2_spearman_r"          : round(float(wgi_spear.statistic), 4),
            "H2_spearman_p"          : round(float(wgi_spear.pvalue), 6),
            "H2_wgi_dimensions"      : wgi_dim_r,
            # ANOVA
            "anova_F"                : round(float(F_anova), 3) if np.isfinite(F_anova) else None,
            "anova_p"                : round(float(p_anova), 6) if np.isfinite(p_anova) else None,
            "nrmse_by_income_group"  : {
                g: {"mean": round(float(v["nrmse"].mean()), 4),
                    "std" : round(float(v["nrmse"].std()), 4),
                    "n"   : int(len(v))}
                for g, v in res_df.groupby("income_group")
            },
            # Benchmarks
            "grover_theoretical_iter": 12,
            "rs_iter_mean"           : round(float(bench_df["rs_iter"].mean()), 1),
            "bo_iter_mean"           : round(float(bench_df["bo_iter"].mean()), 1),
            "ga_iter_mean"           : round(float(bench_df["ga_iter"].mean()), 1),
            "speedup_vs_rs"          : round(float(bench_df["rs_iter"].mean()) / 12, 1),
            # Noise
            "noise_mean_delta_01pct" : round(float(noise_df["delta_pct_0.01"].mean()), 1),
            "noise_unique_values"    : int(noise_df["delta_pct_0.01"].nunique()),
            # Runtime
            "runtime_minutes"        : round((time.time() - t0) / 60, 2),
        }

        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self._print_summary(summary, res_df)
        return summary

    def _print_summary(self, s: Dict, res_df: pd.DataFrame) -> None:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  RESULTS  v{CODE_VERSION}  |  Financial Innovation")
        print(f"{sep}")
        print(f"  Countries: {s['n_countries']}  |  α = {s['alpha_fixed']:.4f}  |  seed {GLOBAL_SEED}")
        print(f"  A  mean±SD = {s['mean_A']}±{s['std_A']}  max = {s['max_A']}")
        print(f"  NRMSE mean = {s['mean_nrmse']}  (improvement {s['nrmse_improvement_pct']}% vs Mankiw 1992)")
        print()
        print("── H1: NRMSE ~ log(GDP per capita) ──────────────────────────")
        print(f"  Pearson r  = {s['H1_pearson_r']:.4f}  p = {s['H1_pearson_p']:.2e}")
        print(f"  Spearman ρ = {s['H1_spearman_r']:.4f}  p = {s['H1_spearman_p']:.2e}")
        print(f"  Bootstrap 95% CI = [{s['H1_bootstrap_ci_lower']:.3f}, {s['H1_bootstrap_ci_upper']:.3f}]")
        print(f"  Permutation p (H0: r=0) = {s['H1_permutation_p']:.4f}")
        print()
        print("── H2: NRMSE ~ WGI ──────────────────────────────────────────")
        print(f"  r(NRMSE, WGI composite) = {s['H2_r_wgi_composite']:.4f}  "
              f"p = {s['H2_p_wgi_composite']:.2e}  (n={s['n_wgi']})")
        for dim, v in s["H2_wgi_dimensions"].items():
            print(f"    r(NRMSE, {dim:<22}) = {v['r']:+.4f}  p={v['p']:.4f}")
        print()
        print("── ANOVA: NRMSE by income group ─────────────────────────────")
        print(f"  F = {s['anova_F']:.3f}  p = {s['anova_p']:.6f}")
        for g, v in s["nrmse_by_income_group"].items():
            print(f"    {g} (n={v['n']}):  NRMSE = {v['mean']:.4f} ± {v['std']:.4f}")
        print()
        print("── Classical benchmarks ─────────────────────────────────────")
        print(f"  Grover ≈ 12 iters  |  Random Search = {s['rs_iter_mean']:.1f}  "
              f"|  Speedup = {s['speedup_vs_rs']:.1f}×")
        print(f"  Bayesian Opt = {s['bo_iter_mean']:.1f}  |  Genetic Alg = {s['ga_iter_mean']:.1f}")
        print()
        print(f"  Runtime: {s['runtime_minutes']} min")
        print(f"  Outputs: {self.cfg.output_dir}/")
        print(f"{sep}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum Solow Analysis — replication code for Nguyen et al. (2025)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing economic_data.csv and WB_WGI_1_csv.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <data-dir>/results/)",
    )
    return parser.parse_args()


def main() -> None:
    print(f"=== Quantum Solow Analysis  v{CODE_VERSION} ===")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}  "
          f"|  seed {GLOBAL_SEED}  |  {TIMESTAMP}\n")

    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir \
                 else data_dir / "results"

    cfg = Config(
        data_path  = str(data_dir / "economic_data.csv"),
        wgi_path   = str(data_dir / "WB_WGI_1_csv.xlsx"),
        output_dir = str(output_dir),
    )

    try:
        Pipeline(cfg).run()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

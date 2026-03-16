# Quantum Solow Analysis — Replication Code

Replication code for:

> Nguyen, Tran Thao; Canh, Tran Quang; Quang, Nguyen Duy; Nghia, Huynh Nhat (2025).
> **"Quantum Optimization of Economic Growth Models: Institutional Insights from Cross-Country Evidence."**
> *Financial Innovation* (Springer, Q1). DOI: [to be assigned]

---

## Overview

This repository implements Grover's quantum search algorithm to optimize savings
rate (`s`) and depreciation rate (`δ`) in the Solow (1956) growth model for
49 countries over 2000–2024. The central finding is that quantum optimization
achieves lower NRMSE in economically developed, institutionally stable countries
(*r* = −0.682, *p* < 0.001), establishing a **Quantum Parameter Identification
(QPI) channel** between governance quality and optimization landscape smoothness.

---

## Repository Structure

```
quantum-solow-analysis/
├── quantum_solow_analysis.py   # Main analysis script (single file)
├── requirements.txt
├── README.md
└── data/                       # Place your data files here (not included)
    ├── economic_data.csv       # WDI panel data (see Data section)
    └── WB_WGI_1_csv.xlsx       # World Bank WGI data (see Data section)
```

---

## Requirements

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.7
openpyxl>=3.1
statsmodels>=0.14
```

---

## Data

The analysis uses two publicly available World Bank datasets.

**1. economic_data.csv** — World Development Indicators (WDI)

Download from: https://databank.worldbank.org/source/world-development-indicators

Required columns:

| Column | Description |
|--------|-------------|
| `Country` | ISO 3166-1 alpha-3 country code |
| `Year` | 2000–2024 |
| `GDP_billions` | GDP in current USD (billions) |
| `Capital_billions` | Gross capital formation, current USD (billions) |
| `Labor_millions` | Total labor force (millions) |

**2. WB_WGI_1_csv.xlsx** — Worldwide Governance Indicators (WGI)

Download from: https://info.worldbank.org/governance/wgi/

Expected long-format columns: `REF_AREA`, `INDICATOR`, `TIME_PERIOD`, `OBS_VALUE`

> **Note:** In the WGI file, rename country code `ROU` → `ROM` to match WDI
> country codes before running the analysis.

---

## Usage

```bash
# Basic usage (data files in ./data/)
python quantum_solow_analysis.py --data-dir data/

# Custom output directory
python quantum_solow_analysis.py --data-dir data/ --output-dir results/

# If data files are in the current directory
python quantum_solow_analysis.py
```

---

## Output Files

All outputs are written to `--output-dir` (default: `<data-dir>/results/`).

| File | Contents |
|------|----------|
| `results_main.csv` | Country-level TFP estimates and Grover NRMSE (Table 1) |
| `results_noise.csv` | NISQ noise sensitivity per country (Table 4) |
| `results_benchmarks.csv` | Classical algorithm comparison (Table 3, Panel A) |
| `results_wgi.csv` | WGI dimension correlations (Table 3, Panel B) |
| `summary.json` | All headline statistics cited in the paper |
| `Figure1_NRMSE_GDP_WGI.png` | Scatter plots NRMSE vs GDP/c and WGI (Figure 1) |

---

## Key Parameters (all in `Config`)

| Parameter | Value | Reference |
|-----------|-------|-----------|
| `ALPHA_FIXED` | 1/3 | Mankiw, Romer & Weil (1992) |
| `NUM_QUBITS` | 8 (256 states) | Section 3.3 |
| `NRMSE_PERCENTILE` | 25th | Oracle threshold, Section 3.3 |
| `CLASSICAL_MAX_ITER` | 200 | Table 3 |
| `BOOTSTRAP_REPS` | 1,000 | Section 3.4 |
| `PERMUTATION_REPS` | 10,000 | Section 3.4 |
| `GLOBAL_SEED` | 42 | Reproducibility |

---

## Reproducing Paper Results

The following values from the paper are produced by `summary.json`:

```
H1: r(NRMSE, log GDP/c)    = −0.682   p < 0.001
    Bootstrap 95% CI        = [−0.815, −0.514]
    Spearman ρ              = −0.629   p < 0.001
    Permutation p           = 0.000

H2: r(NRMSE, WGI composite) = −0.607  p < 0.001  (n=47)
    All 6 WGI dimensions: significant (FDR corrected)

ANOVA: F(2,46) = 12.928, p < 0.001
    High income NRMSE = 0.174 ± 0.061
    Lower-middle NRMSE = 0.357 ± 0.080

Benchmark speedup vs Random Search: 7.5×
NRMSE improvement vs Mankiw (1992): 72.3%
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{nguyen2025quantum,
  title   = {Quantum Optimization of Economic Growth Models:
             Institutional Insights from Cross-Country Evidence},
  author  = {Nguyen, Tran Thao and Canh, Tran Quang and
             Quang, Nguyen Duy and Nghia, Huynh Nhat},
  journal = {Financial Innovation},
  year    = {2025},
  doi     = {[to be assigned]}
}
```

---

## License

MIT License. See `LICENSE` for details.

Data sources (WDI, WGI) are subject to World Bank terms of use.

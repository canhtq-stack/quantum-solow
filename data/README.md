```markdown
# Data Download Instructions

This folder should contain two files required by `quantum_solow_analysis.py`:

1. `economic_data.csv`  
   - Source: World Bank World Development Indicators (WDI)  
   - Variables: Country, Year, GDP_billions, Capital_billions, Labor_millions, Income_Group  
   - How to obtain:  
     Go to https://databank.worldbank.org/source/world-development-indicators  
     Select the variables above (use long format if possible), filter years 2000–2024, select the 49 countries listed in the paper (or all countries and subset later), and export as CSV.

2. `WB_WGI_1_csv.xlsx`  
   - Source: Worldwide Governance Indicators (WGI) – long format  
   - Columns needed: REF_AREA (country code), INDICATOR (e.g. VA.EST, PV.EST, ...), TIME_PERIOD, OBS_VALUE  
   - How to obtain:  
     Go to https://info.worldbank.org/governance/wgi/  
     → Data & Analytics → Download full dataset (Excel)  
     → Use the long-format version or pivot the standard download accordingly.

**Important notes**  
- The code expects exactly these filenames and is case-sensitive.  
- Missing or mismatched columns will cause errors — check the script header for required structure.  
- We do not redistribute the raw data due to World Bank terms of use.

If you encounter issues preparing the data, feel free to open an issue.

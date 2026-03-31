# Metis — D300 Causal Inference and Machine Learning

Replicates the OLS results from Moretti, Steinwender & Van Reenen (2025) and applies Double/Debiased Machine Learning to test functional-form robustness.

## Setup

```
conda env create -f environment.yml
conda activate metis
```

## Data

The raw data comes from the replication package for Moretti, Steinwender & Van Reenen (2025, *Review of Economics and Statistics*), downloaded from the Harvard Dataverse at https://doi.org/10.7910/DVN/0TNZBQ. The package contains OECD source files which `01_build_panel` merges into a single panel:

- `DATA.txt`, `dimCOU.txt`, `dimIND.txt`, `dimVAR.txt` — OECD STAN (industrial production, employment, capital stock)
- `BERD_FUNDS_*.csv` — OECD BERD (business R&D expenditure by funding source)
- `GBAORD_NABS2007_*.csv` — OECD GBAORD (government R&D budget allocations)
- `SNA_TABLE1_*.csv` — OECD National Accounts (GDP)
- `rdtaxcredits.csv` — R&D tax credit/subsidy rates
- `indcorrespondence.dta` — industry classification crosswalk

## Replication

Run the notebooks in order:

1. `01_build_panel` — merges raw OECD data into `data/processed/panel.csv`
2. `02_replicate_ols` — replicates Table 1 Panel A (Cols 1–4)
3. `03_dml_estimation` — DML-Lasso, DML-RF, and CausalForestDML
4. `04_attenuation_investigation` — diagnostics for tree-based attenuation

Figures are saved to `output/`.

## Project structure

```
src/
  panel.py      — sample construction and FE absorption
  ols.py        — WLS estimation with CGM clustered SEs
  inference.py  — cluster-robust variance-covariance matrices
notebooks/      — analysis pipeline (run in order)
data/raw/       — OECD source files
data/processed/ — constructed panel
output/         — figures
```

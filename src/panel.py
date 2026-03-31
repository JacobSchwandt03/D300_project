"""
panel.py — sample construction and panel transformation utilities.
"""

import numpy as np
import pandas as pd


def build_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Tag observations for the Col-1 and Col-2 estimation samples based on
    non-missing required variables."""
    panel = panel.copy()
    panel["col1"] = (
        panel["lnrdexgov"].notna()
        & panel["l1_rdgov"].notna()
        & panel["weight"].notna()
    )
    panel["col2"] = panel["col1"] & (
        panel["l1_indctryprod"].notna()
        & panel["l1_gdp"].notna()
        & panel["l1_taxsubs"].notna()
    )
    return panel


def add_country_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create country × year-trend interactions, omitting the first country
    as reference."""
    df = df.copy()
    countries = sorted(df["ctrynr"].unique())
    trend_cols: list[str] = []
    for c in countries[1:]:  # skip first country as reference
        col = f"ctrend_{c}"
        df[col] = (df["ctrynr"] == c).astype(float) * df["yeartrend"]
        trend_cols.append(col)
    return df, trend_cols


def weighted_demean_twoway(
    data: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    w: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> np.ndarray:
    """Absorb two-way FEs via Gauss-Seidel alternating weighted demeaning
    (the FWL within-transformation for WLS).  Repeat until max change < tol."""
    res = np.array(data, dtype=float)
    g1s = pd.Series(g1)
    g2s = pd.Series(g2)
    ws = pd.Series(w)

    for _ in range(max_iter):
        old = res.copy()
        for gs in (g1s, g2s):
            wsum = ws.groupby(gs).transform("sum").values  # (n,)
            # weighted group mean for each column
            wm = (
                pd.DataFrame(res).multiply(ws, axis=0).groupby(gs).transform("sum").values
            ) / wsum[:, None]
            res = res - wm
        if np.max(np.abs(res - old)) < tol:
            break

    return res

"""
ols.py — WLS with two-way fixed effects and CGM clustered standard errors.
"""

import numpy as np
import pandas as pd

from src.inference import cluster_meat
from src.panel import weighted_demean_twoway


def run_wls_fe(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    fe1_col: str,
    fe2_col: str,
    w_col: str,
    cl1_col: str,
    cl2_col: str,
) -> tuple[float, float, int]:
    """WLS with two-way FE absorption and CGM clustered SEs.
    Returns (beta, se, n) for the first variable in x_cols."""
    y = df[y_col].values.astype(float)
    X = df[x_cols].values.astype(float)
    w = df[w_col].values.astype(float)

    # absorb two-way FE via alternating weighted projections (FWL theorem)
    Xy = np.column_stack([y, X])
    Xy_dem = weighted_demean_twoway(
        Xy,
        df[fe1_col].values,
        df[fe2_col].values,
        w,
    )
    y_dem = Xy_dem[:, 0]
    X_dem = Xy_dem[:, 1:]

    # WLS normal equations: (X'WX)β = X'Wy
    XtW = X_dem.T * w          # (k, n)
    XtWX = XtW @ X_dem         # (k, k)
    XtWy = XtW @ y_dem         # (k,)
    beta, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=1e-10)

    e = y_dem - X_dem @ beta   # WLS residuals

    # Cameron-Gelbach-Miller two-way sandwich
    cl1 = df[cl1_col].values
    cl2 = df[cl2_col].values
    cl12 = np.array([f"{a}_{b}" for a, b in zip(cl1, cl2)])
    bread = np.linalg.pinv(XtWX, rcond=1e-10)
    V = bread @ (
        cluster_meat(X_dem, e, w, cl1)
        + cluster_meat(X_dem, e, w, cl2)
        - cluster_meat(X_dem, e, w, cl12)
    ) @ bread

    return float(beta[0]), float(np.sqrt(np.abs(np.diag(V)))[0]), int(len(y))

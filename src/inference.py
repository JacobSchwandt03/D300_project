"""
inference.py — robust and clustered variance-covariance utilities.
"""

import numpy as np


def cluster_meat(
    X: np.ndarray,
    e: np.ndarray,
    w: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """One-way cluster-robust meat matrix: sum within-cluster scores and
    then accumulate outer products.  Inputs are FE-demeaned."""
    scores = X * (e * w)[:, None]  # (n, k)
    k = X.shape[1]
    meat = np.zeros((k, k))
    for cl in np.unique(cluster_ids):
        s = scores[cluster_ids == cl].sum(axis=0)  # (k,)
        meat += np.outer(s, s)
    return meat


def cgm_twoway(
    X: np.ndarray,
    e: np.ndarray,
    w: np.ndarray,
    cl1: np.ndarray,
    cl2: np.ndarray,
) -> np.ndarray:
    """CGM (2011) two-way clustered sandwich, where
    cl1/cl2 are the two clustering dimensions (country & year)."""
    cl12 = np.array([f"{a}_{b}" for a, b in zip(cl1, cl2)])
    bread = np.linalg.pinv(X.T @ (X * w[:, None]), rcond=1e-10)
    M = (
        cluster_meat(X, e, w, cl1)
        + cluster_meat(X, e, w, cl2)
        - cluster_meat(X, e, w, cl12)
    )
    return bread @ M @ bread

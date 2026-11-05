"""Dependence-aware inference helpers for repeated cross-validation."""

from math import sqrt

import numpy as np
from scipy import stats


def corrected_repeated_cv_summary(values, n_splits: int = 5, confidence: float = 0.95) -> dict:
    """Nadeau-Bengio corrected interval for repeated k-fold CV scores."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("values must contain at least two fold scores")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    correction = 1.0 / len(x) + 1.0 / (n_splits - 1)
    standard_error = sqrt(correction * variance)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(x) - 1))
    return {
        "mean": mean,
        "std": sqrt(variance),
        "standard_error": standard_error,
        "ci_low": mean - critical * standard_error,
        "ci_high": mean + critical * standard_error,
        "correction_factor": correction,
        "n_evaluations": len(x),
        "n_splits": n_splits,
        "confidence": confidence,
    }


def corrected_repeated_cv_test(differences, n_splits: int = 5, confidence: float = 0.95) -> dict:
    """Test paired fold-wise differences using corrected repeated-CV variance."""
    x = np.asarray(differences, dtype=float)
    summary = corrected_repeated_cv_summary(x, n_splits=n_splits, confidence=confidence)
    standard_error = summary["standard_error"]
    if standard_error == 0:
        statistic = 0.0 if summary["mean"] == 0 else float("inf")
        p_value = 1.0 if summary["mean"] == 0 else 0.0
    else:
        statistic = summary["mean"] / standard_error
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=len(x) - 1))
    return {
        **summary,
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": len(x) - 1,
    }

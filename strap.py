"""STRAP: Statistical Testing along a Regularization Path.

Minimal reference implementation of the decision layer. Given, for each of K
regularization strengths, the reconstruction errors of held-out calibration
samples and of test samples, it returns per-strength p-values, the voting
score, the m-of-K decisions and the ranking score.
"""
import numpy as np
from scipy import stats
from scipy.special import boxcox as _boxcox


def calibrate(cal_errors):
    """Fit Box-Cox on calibration errors; return (tau, mean, std, n)."""
    e = np.maximum(np.asarray(cal_errors, dtype=float), 1e-12)
    y, tau = stats.boxcox(e)
    return tau, y.mean(), y.std(ddof=1), len(y)


def pvalues(test_errors, calibration):
    """One-sided prediction-interval p-values (Eq. 5): test ~ t_{n-1} under H0."""
    tau, ybar, s, n = calibration
    y0 = _boxcox(np.maximum(np.asarray(test_errors, dtype=float), 1e-12), tau)
    t = (y0 - ybar) / (s * np.sqrt(1.0 + 1.0 / n))
    return stats.t.sf(t, df=n - 1), t


def strap(cal_errors_per_strength, test_errors_per_strength, alpha=0.05, m=3, bonferroni=False):
    """Run STRAP's scoring phase.

    cal_errors_per_strength / test_errors_per_strength: lists of K arrays
    (index 0 must be the unregularized strength, lambda = 0).
    Returns dict with p-values (K x n_test), votes, flags (m-of-K rule) and the
    ranking score (statistic at lambda = 0).
    """
    K = len(cal_errors_per_strength)
    P, T = zip(*[pvalues(te, calibrate(ce)) for ce, te in zip(cal_errors_per_strength, test_errors_per_strength)])
    P, T = np.stack(P), np.stack(T)
    level = alpha / K if bonferroni else alpha
    reject = P < level
    votes = reject.sum(axis=0)
    flags = reject.any(axis=0) if bonferroni else votes >= m
    return {"pvalues": P, "votes": votes, "flags": flags, "score": T[0]}


if __name__ == "__main__":
    # sanity check: normal test errors from the calibration distribution -> FPR close to alpha
    rng = np.random.default_rng(0)
    cal = [rng.lognormal(-4, 0.5, 2000) for _ in range(5)]
    test = [rng.lognormal(-4, 0.5, 20000) for _ in range(5)]
    out = strap(cal, test, alpha=0.05, m=1)
    print("per-strength FPR:", (out["pvalues"] < 0.05).mean(axis=1).round(3))

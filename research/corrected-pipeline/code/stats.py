"""Interval and paired-test helpers matching the thesis conventions."""
import math
import numpy as np

def wilson(k, n, z=1.959963984540054):
    if n <= 0: return (0.0, 0.0)
    p = k/n; den = 1 + z*z/n; c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0.0, c-h), min(1.0, c+h)

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k+1)) / 2**n
    return min(1.0, 2*p)

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0.0]*m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r)*ps[i])); adj[i] = run
    return adj

def paired_diff_boot(a, b, elig, clusters, draws=10000, seed=20261029):
    """Difference in accuracy (a minus b) over eligible items, resampling clusters (receipts) with replacement."""
    a = np.asarray(a, float); b = np.asarray(b, float); e = np.asarray(elig, float); cl = np.asarray(clusters)
    ids = sorted(set(cl)); idx = {c: np.where(cl == c)[0] for c in ids}
    A = np.array([(a*e)[idx[c]].sum() for c in ids]); B = np.array([(b*e)[idx[c]].sum() for c in ids]); E = np.array([e[idx[c]].sum() for c in ids])
    rng = np.random.default_rng(seed); s = rng.integers(0, len(ids), size=(draws, len(ids)))
    den = E[s].sum(1); d = np.divide(A[s].sum(1) - B[s].sum(1), den, out=np.zeros(draws), where=den > 0)
    est = (A.sum() - B.sum())/E.sum()
    lo, hi = np.quantile(d, [.025, .975]); return est*100, lo*100, hi*100

def f1_counts(c):
    tp, fp, fn = c
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0

def paired_f1_boot(ca, cb, clusters, draws=10000, seed=20261029):
    """Row-F1 difference (a minus b) with cluster bootstrap; ca/cb are per-item (tp, fp, fn)."""
    ca = np.asarray(ca, float); cb = np.asarray(cb, float); cl = np.asarray(clusters)
    ids = sorted(set(cl)); idx = {c: np.where(cl == c)[0] for c in ids}
    A = np.array([ca[idx[c]].sum(0) for c in ids]); B = np.array([cb[idx[c]].sum(0) for c in ids])
    rng = np.random.default_rng(seed); s = rng.integers(0, len(ids), size=(draws, len(ids)))
    def f(M):
        S = M[s].sum(1); den = 2*S[:, 0] + S[:, 1] + S[:, 2]
        return np.divide(2*S[:, 0], den, out=np.zeros(draws), where=den > 0)
    d = f(A) - f(B); lo, hi = np.quantile(d, [.025, .975])
    return f1_counts(A.sum(0)) - f1_counts(B.sum(0)), lo, hi

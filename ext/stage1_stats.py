"""Stage 1 statistics (protocol v1 §5, amendment 1 A5/A6): paired geometric gain, win fraction, pointwise percentile
bootstrap over operators (10 000 resamples, rng 20261016), P1 bands in the frozen order."""
import numpy as np

B_RESAMPLES, B_SEED = 10_000, 20261016


def paired(e_x, e_y, ids=None):
    """gain of Y over X: G = exp(mean log(e_x / e_y)); W = mean(e_y < e_x); 95% pointwise bootstrap interval of G.
    Complete cases only (both finite and > 0); excluded ids are returned."""
    e_x, e_y = np.asarray(e_x, float), np.asarray(e_y, float)
    ok = np.isfinite(e_x) & np.isfinite(e_y) & (e_x > 0) & (e_y > 0)
    lr = np.log(e_x[ok] / e_y[ok]); n = len(lr)
    rng = np.random.default_rng(B_SEED); idx = rng.integers(0, n, size=(B_RESAMPLES, n))
    bs = np.exp(lr[idx].mean(1))
    return dict(G=float(np.exp(lr.mean())), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                W=float(np.mean(e_y[ok] < e_x[ok])), n=int(n),
                excluded=[] if ids is None else [int(i) for i, o in zip(ids, ok) if not o])


def band(R):
    """P1 band for R = G(PCG -> OA-VCB) with its interval, frozen order (amendment of v1 §5)."""
    if R['hi'] < 1: return 'classical adaptive smoothing more accurate'
    if R['hi'] <= 1.25: return 'largely accounted for at matched stages'
    if R['lo'] >= 2: return 'not accounted for by Krylov smoothing'
    return 'partly accounted for / inconclusive'


def summary(e):
    e = np.asarray(e, float)
    return dict(mean=float(np.mean(e)), median=float(np.median(e)), p95=float(np.quantile(e, .95)), max=float(e.max()))


def phi(e_ref, e_y, e_oa, ids=None):
    """recovered log-gain phi_Y = log G(ref -> Y) / log G(ref -> OA-VCB) (v1 §5), descriptive; bootstrap over operators with
    the same resample indices for numerator and denominator.  Flag 'unstable' if the interval of G(ref -> OA-VCB) reaches
    below 1.25."""
    e_ref, e_y, e_oa = (np.asarray(v, float) for v in (e_ref, e_y, e_oa))
    ok = np.all([np.isfinite(v) & (v > 0) for v in (e_ref, e_y, e_oa)], 0)
    a = np.log(e_ref[ok] / e_y[ok]); b = np.log(e_ref[ok] / e_oa[ok]); n = len(a)
    idx = np.random.default_rng(B_SEED).integers(0, n, size=(B_RESAMPLES, n))
    bs = a[idx].mean(1) / b[idx].mean(1); gO = np.exp(b[idx].mean(1))
    return dict(phi=float(a.mean() / b.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), n=int(n),
                unstable=bool(np.quantile(gO, 0.025) < 1.25),
                excluded=[] if ids is None else [int(i) for i, o in zip(ids, ok) if not o])


def guarded(e, e_uc):
    """failure assigned the error of u_c (v1 §8 / A6 guarded variant); returns (errors, number of fallbacks)."""
    e, e_uc = np.asarray(e, float), np.asarray(e_uc, float); bad = ~np.isfinite(e)
    return np.where(bad, e_uc, e), int(bad.sum())


def paired_clustered(e_x, e_y, clusters):
    """as paired(), several samples per operator (set LC): G over all samples; bootstrap resamples OPERATORS (clusters),
    each drawn operator contributing all its samples (v1 §5)."""
    e_x, e_y, cl = np.asarray(e_x, float), np.asarray(e_y, float), np.asarray(clusters)
    ok = np.isfinite(e_x) & np.isfinite(e_y) & (e_x > 0) & (e_y > 0)
    lr = np.log(e_x[ok] / e_y[ok]); c = cl[ok]; u = np.unique(c)
    s = np.array([lr[c == k].sum() for k in u]); m = np.array([(c == k).sum() for k in u])
    idx = np.random.default_rng(B_SEED).integers(0, len(u), size=(B_RESAMPLES, len(u)))
    bs = np.exp(s[idx].sum(1) / m[idx].sum(1))
    return dict(G=float(np.exp(lr.mean())), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                W=float(np.mean(e_y[ok] < e_x[ok])), n=int(ok.sum()), n_operators=int(len(u)), excluded=int((~ok).sum()))

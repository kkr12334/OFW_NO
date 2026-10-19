"""Load vectors (copied arithmetic from diagnostics_cost/rhs_random_model.py, which mirrors train_oa_layer.load_vec).
random_load(H, L, seed): the random-load training distribution with generator seed `seed`
  (training: seed = 777_000 + 100_000 * DS + i;  set LC: seed = 2026101622 * 10**3 + i).
lc_loads(H, L, i): set LC loads (protocol v1 §4): f = -1, smooth +-, pure localised Gaussian (width 0.03) from rng
  2026101621 * 10**3 + i (same draw order as rhs_random_model.loads), and a random-load draw from rng 2026101622 * 10**3 + i."""
import numpy as np


class Geo:
    def __init__(self, H, L):
        self.H, self.L = H, L; xy = H.coords[H.elem]
        self.area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
        self.cx, self.cy = xy.mean(1).T

    def assemble(self, fe):
        H = self.H; Fv = np.zeros(len(H.coords)); m = H.keep
        for j in range(3): np.add.at(Fv, H.elem[m, j], (fe * self.area)[m] / 3.0)
        return Fv[H.nodes[self.L]]


def random_load(G, seed):
    cx, cy = G.cx, G.cy; g = np.random.default_rng(seed); K = g.normal(size=(4, 4))
    fe = g.uniform(-1, 1) + sum(K[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) / (1 + p + q) for p in range(4) for q in range(4))
    if g.uniform() < 0.5:
        x0 = g.uniform(0.1, 0.9, 2); w = g.uniform(0.02, 0.08)
        fe = fe + g.uniform(2, 10) * np.exp(-((cx - x0[0]) ** 2 + (cy - x0[1]) ** 2) / (2 * w ** 2))
    return G.assemble(fe)


def lc_loads(G, i):
    cx, cy = G.cx, G.cy; h = np.random.default_rng(2026101621 * 10 ** 3 + int(i)); K2 = h.normal(size=(4, 4))
    sm = sum(K2[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) for p in range(4) for q in range(4))
    c0 = h.uniform(0.15, 0.85, 2); pt = np.exp(-((cx - c0[0]) ** 2 + (cy - c0[1]) ** 2) / (2 * 0.03 ** 2))
    return {'f = -1': -G.H.F, 'smooth +-': G.assemble(sm), 'point source (w 0.03)': G.assemble(pt),
            'random-load draw': random_load(G, 2026101622 * 10 ** 3 + int(i))}


def random_model_scale(S, DS):
    """the random-load model's own input scale: rms of U - U_c over training samples 0-299 with their random loads."""
    from scipy.sparse.linalg import spsolve
    G = Geo(S['H'], S['L']); P = S['P']; PT = P.T.tocsr(); D = []
    for i in S['tr']:
        A = S['A'][i]; F = random_load(G, 777_000 + 100_000 * DS + int(i)); u = spsolve(A.tocsc(), F)
        D.append(u - P @ spsolve((PT @ A @ P).tocsc(), PT @ F))
    return float(np.sqrt(np.mean(np.array(D) ** 2)))

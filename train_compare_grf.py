"""GRF-coefficient version of train_compare.py -- also missing from the
shared folder. Mirrors prepare()/run() there as closely as possible, with
the two things that MUST differ for a GRF (no finite mu):

  1. OFW-NO (the dense mu->coefficient model) is not definable and is
     dropped. Only OFW-NO-sp, FNO, WNO, MLP run.
  2. Capacity matching can no longer anchor on OFW-NO's param count. It
     anchors on OFW-NO-sp's instead (fixed hidden width, matching the
     channelized run's spatial-encoder size), and -- this is the fix for
     the bug found in ofwno_benchmark_v2.match_capacity() -- MLP's budget
     is computed against its ACTUAL input dimension (the flattened grid,
     n_grid*n_grid), not the hardcoded mu=16 that match_capacity() silently
     assumes. If that hardcoding were left in for a GRF run, MLP would be
     sized as if it had a 16-dim input while actually being built with a
     ~1000+-dim one -- a real parameter-count leak in the *wrong* model's
     favour that's worth measuring, not just asserting.
"""
import sys, time
import numpy as np
import torch, torch.nn as nn

from ofwno_benchmark_v2 import fno_params, wno_params, mlp_params
from ofwno_nets import (RealFNO2D, RealWNO2D, MLPBaseline, SpatialOFWNO,
                        spatial_ofwno_params, build_grid_bijection,
                        grid_to_nodal, _WAVELET_BACKEND)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def real_params(m):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in m.parameters() if p.requires_grad)


def match_capacity_grf(level_dims, mu_dim, sp_hidden=32, modes=12, layers=4,
                       head_hidden=64, verbose=True):
    """FNO/WNO/MLP sized to OFW-NO-sp's OWN parameter count (no dense OFW-NO
    exists for a GRF to anchor on instead). Unlike match_capacity(), mu_dim
    -- the true MLP input width -- is a required argument, not a hardcoded 16.
    """
    dims = list(level_dims.values()) if isinstance(level_dims, dict) else list(level_dims)
    target = spatial_ofwno_params(sp_hidden, modes, layers, head_hidden, len(dims))
    n_out = sum(dims)

    def bisect(f, lo, hi):
        while lo < hi:
            m = (lo + hi) // 2
            if f(m) < target: lo = m + 1
            else: hi = m
        return lo

    h_fno = bisect(lambda h: fno_params(h, modes, layers), 4, 512)
    h_wno = bisect(lambda h: wno_params(h, layers), 4, 512)
    h_mlp = bisect(lambda h: mlp_params(mu_dim, n_out, (256, 256, h)), 4, 65536)
    cfg = dict(target=target, sp_hidden=sp_hidden, fno_modes=modes,
              fno_hidden=h_fno, fno_layers=layers, wno_hidden=h_wno, wno_layers=layers,
              mlp_hidden=(256, 256, h_mlp))
    if verbose:
        print(f"capacity target (OFW-NO-sp, hidden={sp_hidden}): {target:,} params  "
              f"(mu_dim used for MLP budget: {mu_dim})")
        for nm, p in (("FNO", fno_params(h_fno, modes, layers)),
                      ("WNO", wno_params(h_wno, layers)),
                      ("MLP", mlp_params(mu_dim, n_out, (256, 256, h_mlp)))):
            print(f"  {nm}: {p:>9,}  ({p/target:.2f}x target)")
        print(f"  -> fno_hidden={h_fno}, wno_hidden={h_wno}, mlp_hidden=(256,256,{h_mlp})")
    return cfg


def prepare_grf(pool, test_frac=0.25, seed=0):
    mesh, S = pool['mesh'], pool['solver']
    g2n, n2g, n, gxy = build_grid_bijection(mesh, S.max_level)
    assert n == pool['n_grid'], "GRF field grid and mesh grid must coincide"
    G, u = pool['grid_a'], pool['u']
    N = len(G)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = max(2, int(round(test_frac * N)))
    te, tr = idx[:n_te], idx[n_te:]

    # --- shared input encoding: standardised log-field, EXACTLY like the
    # channelized run standardises log(a) -- just over the whole grid instead
    # of 16 channel amplitudes. This is what MLP sees directly (flattened)
    # and what FNO/WNO/OFW-NO-sp see as the field channel of (field, x, y).
    lf = np.log(G)
    m_, s_ = lf[tr].mean(), lf[tr].std() + 1e-12
    field = ((lf - m_) / s_).astype(np.float32)          # (N, n, n)
    mu = field.reshape(N, -1)                              # (N, n*n) -- MLP input

    xy = np.broadcast_to(gxy, (N, n, n, 2))
    Xg = np.concatenate([field[..., None], xy], axis=-1).astype(np.float32)

    bmask = np.zeros(u.shape[1], bool); bmask[list(mesh.get_boundary_nodes())] = True
    u = u.copy(); u[:, bmask] = 0.0
    u_s = u[tr].std() + 1e-12
    levels = sorted(pool['c'])
    c = {l: pool['c'][l].astype(np.float32) for l in levels}
    c_s = {l: (float(c[l][tr].std()) or 1.0) for l in levels}

    R = np.stack([S.to_nodal(e) for e in np.eye(S.ndof)], axis=1).astype(np.float32)
    R[bmask, :] = 0.0
    dead = [l for l in levels if c[l][tr].std() == 0.0]
    for l in dead:
        a_, b_ = S.off[l]; R[:, a_:b_] = 0.0

    lng = {l: n2g[np.asarray(mesh.level_nodes[l], dtype=int)] for l in levels}

    return dict(mu=mu, Xg=Xg, u=u.astype(np.float32), u_s=float(u_s),
                c=c, c_s=c_s, levels=levels, tr=tr, te=te, n=n, g2n=g2n,
                n2g=n2g, bmask=bmask, solver=S, R=R, lng=lng, dead=dead,
                level_dims={l: c[l].shape[1] for l in levels})


def rel_l2(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=1) /
                         (np.linalg.norm(true, axis=1) + 1e-30)))


def _fit(model, batches, n_batch, epochs, lr, tag, verbose):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for b in range(n_batch):
            opt.zero_grad(); loss = batches(model, b); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sch.step()
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    {tag} ep {ep+1}/{epochs}  loss {tot/n_batch:.3e}")
    return time.time() - t0


def _loader(n, bs, seed):
    g = np.random.default_rng(seed)
    def epoch_idx():
        p = g.permutation(n)
        return [p[i:i+bs] for i in range(0, n, bs)]
    return epoch_idx


def run_grf(pool, epochs=150, lr=1e-3, batch=16, seed=0, sp_hidden=32, verbose=True,
           models=('OFW-NO-sp', 'FNO', 'WNO', 'MLP')):
    D = prepare_grf(pool, seed=seed)
    tr, te, S = D['tr'], D['te'], D['solver']
    mu_dim = D['mu'].shape[1]
    cfg = match_capacity_grf(D['level_dims'], mu_dim, sp_hidden=sp_hidden, verbose=verbose)
    if verbose:
        print(f"\ndevice {DEV}   wavelet backend: {_WAVELET_BACKEND}")
        print(f"train {len(tr)}  test {len(te)}  grid {D['n']}x{D['n']}  "
              f"nodes {D['u'].shape[1]}  mlp input dim {mu_dim}\n")

    mu_t = torch.tensor(D['mu'], device=DEV)
    Xg_t = torch.tensor(D['Xg'], device=DEV)
    u_t  = torch.tensor(D['u'] / D['u_s'], device=DEV)
    from ofwno_nets import nodal_to_grid
    ug_t = torch.tensor(nodal_to_grid(D['u'] / D['u_s'], D['g2n']), device=DEV)
    nb   = int(np.ceil(len(tr) / batch))
    out  = {}

    def bench(name, model, step, predict):
        torch.manual_seed(seed)
        p = real_params(model)
        t = _fit(model, step, nb, epochs, lr, name, verbose)
        model.eval()
        with torch.no_grad():
            pr = predict(te)
        pr[:, D['bmask']] = 0.0
        e = rel_l2(pr, D['u'][te])
        out[name] = dict(rel_l2=e, params=p, train_s=t)
        if verbose:
            print(f"  {name:<8} rel-L2 {e:.4e}   {p:,} params   {t:.0f}s train\n")

    idx = _loader(len(tr), batch, seed)
    order = {m: idx() for m in models}

    if 'OFW-NO-sp' in models:
        ln = {l: torch.tensor(D['lng'][l].astype(np.int64), device=DEV) for l in D['levels']}
        m = SpatialOFWNO(ln, 3, sp_hidden, cfg['fno_modes'], cfg['fno_layers'], 64).to(DEV)
        bt = order['OFW-NO-sp']
        R_t = torch.tensor(D['R'] / D['u_s'], device=DEV)
        sc  = torch.tensor(np.concatenate([np.full(D['level_dims'][l], D['c_s'][l])
                                           for l in D['levels']]).astype(np.float32), device=DEV)
        def to_nodal_sp(o):
            return (torch.cat([o[l] for l in D['levels']], dim=-1) * sc) @ R_t.T
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(to_nodal_sp(mod(Xg_t[j])), u_t[j])
        def predict(ix):
            return to_nodal_sp(m(Xg_t[ix])).cpu().numpy() * D['u_s']
        bench('OFW-NO-sp', m, step, predict)

    if 'FNO' in models:
        m = RealFNO2D(3, cfg['fno_hidden'], cfg['fno_modes'], cfg['fno_layers']).to(DEV)
        bt = order['FNO']
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(mod(Xg_t[j]), ug_t[j])
        def predict(ix):
            return grid_to_nodal(m(Xg_t[ix]).cpu().numpy(), D['n2g']) * D['u_s']
        bench('FNO', m, step, predict)

    if 'WNO' in models:
        m = RealWNO2D(3, cfg['wno_hidden'], cfg['wno_layers'], 'db4', 2).to(DEV)
        bt = order['WNO']
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(mod(Xg_t[j]), ug_t[j])
        def predict(ix):
            return grid_to_nodal(m(Xg_t[ix]).cpu().numpy(), D['n2g']) * D['u_s']
        bench('WNO', m, step, predict)

    if 'MLP' in models:
        m = MLPBaseline(mu_dim, D['u'].shape[1], cfg['mlp_hidden']).to(DEV)
        bt = order['MLP']
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(mod(mu_t[j]), u_t[j])
        def predict(ix):
            return m(mu_t[ix]).cpu().numpy() * D['u_s']
        bench('MLP', m, step, predict)

    if verbose:
        print(f"{'model':<12}{'rel-L2 (nodal)':>16}{'params':>12}{'train s':>10}")
        for k, v in sorted(out.items(), key=lambda kv: kv[1]['rel_l2']):
            print(f"{k:<12}{v['rel_l2']:>16.4e}{v['params']:>12,}{v['train_s']:>10.0f}")
    return out


if __name__ == '__main__':
    from ofwno_benchmark_grf import build_pool_grf
    lvl  = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N    = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    eps  = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    pool = build_pool_grf(MAX_LEVEL=lvl, N=N, contrast=1e4, corr_length=0.05, nu=0.5,
                          mode='per_sample', n_layers=6, verbose=True)
    run_grf(pool, epochs=eps)

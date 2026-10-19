"""Capacity-matched baseline comparison for OFW-NO.

Every model sees the SAME samples, the SAME input encoding (standardised
log a), the SAME optimiser/epochs/batch size/seed, and is scored by the SAME
metric: relative L2 error of the NODAL solution on held-out samples.  Only the
architecture and the output representation differ.

  OFW-NO  mu -> per-level wavelet coefficients -> solver.to_nodal (FIXED linear
          map; no per-sample basis build at inference)
  FNO     (a_field, x, y) grid -> nodal grid          (exact bijection, no interp)
  WNO     same input/output as FNO, DWT layers instead of spectral layers
  MLP     mu -> nodal vector directly

Run:  python train_compare.py            # level 4 smoke test, ~2 min CPU
      python train_compare.py 6 300 400  # level, n_samples, epochs
"""
import sys, time
import numpy as np
import torch, torch.nn as nn

from ofwno_benchmark_v2 import build_pool, match_capacity, stack
from ofwno_nets import (LinearSkipOFWNeuralOperator, RealFNO2D, RealWNO2D,
                        MLPBaseline, SpatialOFWNO, spatial_ofwno_params,
                        build_grid_bijection, nodal_to_grid,
                        grid_to_nodal, build_a_field_batch, _WAVELET_BACKEND)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def real_params(m):
    """Complex parameters carry 2 real DOF each; torch's numel() reports 1."""
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
def prepare(pool, n_cx=4, n_cy=4, a0=1.0, test_frac=0.25, seed=0):
    mesh, S = pool['mesh'], pool['solver']
    g2n, n2g, n, gxy = build_grid_bijection(mesh, S.max_level)
    a, u = pool['a'], pool['u']
    N = len(a)
    grf = bool(pool.get('grf', False))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = max(2, int(round(test_frac * N)))
    te, tr = idx[:n_te], idx[n_te:]

    # --- shared input encoding: standardised log a -------------------------
    # Channelized: the 16-dim mu.  GRF: there is no finite mu, so the vector
    # encoding is the whole log-field, flattened (this is the MLP's input;
    # FNO/WNO/SpatialOFWNO see the same field with its grid structure intact).
    la = np.log(pool['a_grid'].reshape(N, -1)) if grf else np.log(a)
    m_, s_ = la[tr].mean(0), la[tr].std(0) + 1e-12
    mu = ((la - m_) / s_).astype(np.float32)

    # --- grid input for FNO/WNO: (log a-field, x, y), a-field standardised --
    af = np.log(pool['a_grid']) if grf else \
         np.log(build_a_field_batch(a, gxy, n_cx, n_cy, a0))
    af = (af - af[tr].mean()) / (af[tr].std() + 1e-12)
    xy = np.broadcast_to(gxy, (N, n, n, 2))
    Xg = np.concatenate([af[..., None], xy], axis=-1).astype(np.float32)

    # --- targets -----------------------------------------------------------
    bmask = np.zeros(u.shape[1], bool); bmask[list(mesh.get_boundary_nodes())] = True
    u = u.copy(); u[:, bmask] = 0.0                    # BC is known to every model
    u_s = u[tr].std() + 1e-12
    levels = sorted(pool['c'])
    c = {l: pool['c'][l].astype(np.float32) for l in levels}
    # a level whose coefficients are identically zero (level 0 is all boundary
    # DOFs) must not be divided by ~0; carry it at unit scale instead.
    c_s = {l: (float(c[l][tr].std()) or 1.0) for l in levels}

    # --- exact reconstruction matrix: u_nodal = R @ c  ----------------------
    # to_nodal is a FIXED linear map (the whole point of expressing the answer
    # in the frozen basis), so materialise it once and put it inside the loss.
    # OFW-NO is then optimised on the same nodal objective every baseline uses,
    # instead of on a per-level coefficient surrogate.
    R = np.stack([S.to_nodal(e) for e in np.eye(S.ndof)], axis=1).astype(np.float32)
    R[bmask, :] = 0.0

    # A level whose coefficients are identically zero (level 0 is entirely
    # boundary DOFs) carries no information, but its basis functions have the
    # LARGEST reconstruction gain: ||R col|| ~ 5 against ~1e-3 for the levels
    # that matter, i.e. ~4000x once the coefficient scales are folded in.  Left
    # live, the network's initial noise on those few outputs swamps the nodal
    # loss and it spends training driving them to exactly zero.  Their true
    # contribution is 0, so zeroing the columns is exact, not an approximation.
    dead = [l for l in levels if c[l][tr].std() == 0.0]
    for l in dead:
        a_, b_ = S.off[l]; R[:, a_:b_] = 0.0

    # grid position of every coefficient, level by level, in coefficient order
    lng = {l: n2g[np.asarray(mesh.level_nodes[l], dtype=int)] for l in levels}

    return dict(mu=mu, Xg=Xg, u=u.astype(np.float32), u_s=float(u_s),
                c=c, c_s=c_s, levels=levels, tr=tr, te=te, n=n, g2n=g2n,
                n2g=n2g, bmask=bmask, solver=S, R=R, lng=lng, dead=dead, grf=grf,
                level_dims={l: c[l].shape[1] for l in levels})


def rel_l2(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=1) /
                         (np.linalg.norm(true, axis=1) + 1e-30)))


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def run(pool, epochs=300, lr=1e-3, batch=16, seed=0, verbose=True,
        models=('OFW-NO', 'OFW-NO-sp', 'FNO', 'WNO', 'MLP')):
    D = prepare(pool, seed=seed)
    tr, te, S = D['tr'], D['te'], D['solver']
    dims = [D['level_dims'][l] for l in D['levels']]
    if D['grf']:
        # no finite mu, so the dense OFW-NO is not defined on this problem;
        # SpatialOFWNO sets the budget and the MLP takes the whole field
        models = tuple(m for m in models if m != 'OFW-NO')
        tgt = spatial_ofwno_params(24, 12, 4, 64, len(D['levels']))
        cfg = match_capacity(dims, verbose=verbose, mu_dim=D['mu'].shape[1], target=tgt)
        cfg['ofw_sp_hidden'] = 24
    else:
        cfg = match_capacity(dims, verbose=verbose)
    if verbose:
        print(f"\ndevice {DEV}   wavelet backend: {_WAVELET_BACKEND}")
        print(f"train {len(tr)}  test {len(te)}  grid {D['n']}x{D['n']}  "
              f"nodes {D['u'].shape[1]}  coeff dim {sum(dims)}\n")

    mu_t = torch.tensor(D['mu'], device=DEV)
    Xg_t = torch.tensor(D['Xg'], device=DEV)
    u_t  = torch.tensor(D['u'] / D['u_s'], device=DEV)
    ug_t = torch.tensor(nodal_to_grid(D['u'] / D['u_s'], D['g2n']), device=DEV)
    c_t  = {l: torch.tensor(D['c'][l] / D['c_s'][l], device=DEV) for l in D['levels']}
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
    order = {m: idx() for m in models}          # identical mini-batch schedule

    # ---- OFW-NO ----
    if 'OFW-NO' in models:
        m = LinearSkipOFWNeuralOperator(D['mu'].shape[1], D['level_dims'],
                                        hidden=64, proj_dim=16).to(DEV)
        m.warm_start_linear(D['mu'][tr], {l: D['c'][l][tr] / D['c_s'][l]
                                          for l in D['levels']})
        m.shrink_nonlinear_branch()
        bt = order['OFW-NO']
        R_t = torch.tensor(D['R'] / D['u_s'], device=DEV)
        sc  = torch.tensor(np.concatenate([np.full(D['level_dims'][l], D['c_s'][l])
                                           for l in D['levels']]).astype(np.float32),
                           device=DEV)
        def to_nodal_t(o):
            c = torch.cat([o[l] for l in D['levels']], dim=-1) * sc
            return c @ R_t.T                       # (B, n_nodes), standardised by u_s
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(to_nodal_t(mod(mu_t[j])), u_t[j])
        def predict(ix):
            return to_nodal_t(m(mu_t[ix])).cpu().numpy() * D['u_s']
        bench('OFW-NO', m, step, predict)

    # ---- OFW-NO with a spatial encoder ----
    if 'OFW-NO-sp' in models:
        if 'ofw_sp_hidden' in cfg:
            lo = cfg['ofw_sp_hidden']
        else:
            target = cfg['target']
            lo, hi = 4, 256
            while lo < hi:
                mid = (lo + hi) // 2
                if spatial_ofwno_params(mid, cfg['fno_modes'], cfg['fno_layers'], 64,
                                        len(D['levels'])) < target: lo = mid + 1
                else: hi = mid
        ln = {l: torch.tensor(D['lng'][l].astype(np.int64), device=DEV)
              for l in D['levels']}
        m = SpatialOFWNO(ln, 3, lo, cfg['fno_modes'], cfg['fno_layers'], 64).to(DEV)
        bt = order['OFW-NO-sp']
        R_t = torch.tensor(D['R'] / D['u_s'], device=DEV)
        sc  = torch.tensor(np.concatenate([np.full(D['level_dims'][l], D['c_s'][l])
                                           for l in D['levels']]).astype(np.float32),
                           device=DEV)
        def to_nodal_sp(o):
            return (torch.cat([o[l] for l in D['levels']], dim=-1) * sc) @ R_t.T
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(to_nodal_sp(mod(Xg_t[j])), u_t[j])
        def predict(ix):
            return to_nodal_sp(m(Xg_t[ix])).cpu().numpy() * D['u_s']
        bench('OFW-NO-sp', m, step, predict)

    # ---- FNO ----
    if 'FNO' in models:
        m = RealFNO2D(3, cfg['fno_hidden'], cfg['fno_modes'], cfg['fno_layers']).to(DEV)
        bt = order['FNO']
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(mod(Xg_t[j]), ug_t[j])
        def predict(ix):
            return grid_to_nodal(m(Xg_t[ix]).cpu().numpy(), D['n2g']) * D['u_s']
        bench('FNO', m, step, predict)

    # ---- WNO ----
    if 'WNO' in models:
        m = RealWNO2D(3, cfg['wno_hidden'], cfg['wno_layers'], 'db4', 2).to(DEV)
        bt = order['WNO']
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(mod(Xg_t[j]), ug_t[j])
        def predict(ix):
            return grid_to_nodal(m(Xg_t[ix]).cpu().numpy(), D['n2g']) * D['u_s']
        bench('WNO', m, step, predict)

    # ---- MLP ----
    if 'MLP' in models:
        m = MLPBaseline(D['mu'].shape[1], D['u'].shape[1], cfg['mlp_hidden']).to(DEV)
        bt = order['MLP']
        def step(mod, b):
            j = tr[bt[b % len(bt)]]
            return nn.functional.mse_loss(mod(mu_t[j]), u_t[j])
        def predict(ix):
            return m(mu_t[ix]).cpu().numpy() * D['u_s']
        bench('MLP', m, step, predict)

    if verbose:
        print(f"{'model':<10}{'rel-L2 (nodal)':>16}{'params':>12}{'train s':>10}")
        for k, v in sorted(out.items(), key=lambda kv: kv[1]['rel_l2']):
            print(f"{k:<10}{v['rel_l2']:>16.4e}{v['params']:>12,}{v['train_s']:>10.0f}")
    return out


if __name__ == '__main__':
    lvl  = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N    = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    eps  = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    field = sys.argv[4] if len(sys.argv) > 4 else 'channel'
    if field == 'grf':
        from ofwno_benchmark_v2 import build_pool_grf
        pool = build_pool_grf(MAX_LEVEL=lvl, N=N, contrast=1e4, corr_length=0.05,
                              nu=0.5, mode='per_sample', n_layers=6, verbose=True)
    else:
        pool = build_pool(MAX_LEVEL=lvl, N=N, contrast=1e4, mode='per_sample',
                          n_layers=6, verbose=True)
    run(pool, epochs=eps)

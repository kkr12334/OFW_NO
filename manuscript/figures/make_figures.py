"""Figures of the paper (11 Oct 2026).  Run from OFW_NO:  python manuscript/figures/make_figures.py
fig_setup.pdf     meshes and coefficient samples (perforated square T0 and L3, channels, barriers with floating islands,
                  polygon T0, elasticity bands)
fig_errormaps.pdf element-wise energy-error density of u_c, the Chebyshev reference and OA-VCB on a median channel sample
                  and a median island barrier sample (held-out seed 1; checkpoints 40ce3ac1f9 / 8266a233d9)
fig_results.pdf   (a) per-sample energy errors, reference vs OA-VCB (channels and barriers, held-out seeds 1-3);
                  (b) two-grid PCG energy error vs iterations (median and quartiles) with the one-pass errors
Data: diagnostics_cost/jacobi_member_*.json (reference arrays, reproduced from the stored records), the stored OA-VCB
records (via make_tables' loader rules) and the checkpoints."""
import os, sys, json, glob, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.colors import LogNorm
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'diagnostics_cost'))
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import coarse_meshes as cm
from oa_lifting import NestedHierarchy, field_channels
from s1_common import setup, load_net, apply
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#8a8984'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'pdf.fonttype': 42})
CK = {'ch': 'resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt',
      'ba': 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt'}
def ckpath(rel):
    for base in (ROOT, '/mnt/user-data/uploads/OA_NO/Working codes/OFW_NO'):
        if os.path.exists(os.path.join(base, rel)): return os.path.join(base, rel)
    hits = glob.glob(os.path.join(ROOT, '**', os.path.basename(rel)), recursive=True)
    if hits: return hits[0]
    raise FileNotFoundError(rel)

def tri_of(H): return mtri.Triangulation(H.coords[:, 0], H.coords[:, 1], H.elem[H.keep])
def mesh_ax(ax, coords, elem, lw=0.4, color=INK):
    ax.triplot(mtri.Triangulation(coords[:, 0], coords[:, 1], elem), lw=lw, color=color)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
def clean(ax): ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]

def islands(H, a):
    """conductive components (a >= 0.5) touching no Dirichlet node -> element mask"""
    L = H.L; dirn = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L])
    cond = np.nonzero(H.keep & (a >= 0.5))[0]; el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M @ M.T).tocsr(); n_, lab = connected_components(G, directed=False)
    touch = np.zeros(n_, bool)
    for c in range(n_):
        if np.isin(el[lab == c].ravel(), dirn).any(): touch[c] = True
    mask = np.zeros(H.n_elem, bool); mask[cond[~touch[lab]]] = True
    return mask

# ------------------------------------------------------------------ Figure: setup
def fig_setup():
    T0 = cm.perforated_square(n=9)
    P0 = cm.polygon_domain([(0, 0), (1, 0), (1, .4), (.6, .6), (1, .8), (1, 1), (0, 1)], n=9)
    H = NestedHierarchy(3, coarse=T0)
    rng = np.random.default_rng(1001); a_ch = field_channels(H, 1e4, rng)
    rng = np.random.default_rng(1001); a_ba = field_channels(H, 1e-4, rng)
    # find a barrier sample of seed 1 with islands (first in the test range)
    rng = np.random.default_rng(1001); fields = [field_channels(H, 1e-4, rng) for _ in range(400)]
    ib = None
    for i in range(300, 400):                                        # 11 Oct (Astra): TEST sample, disclosed with its ID
        isl = islands(H, fields[i])
        if isl.sum() > 30: a_ba = fields[i]; ib = i; break
    outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
    He = NestedHierarchy(3, coarse=T0, bc='mixed', neumann=outer); rng = np.random.default_rng(2000); E = field_channels(He, 1e2, rng)
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.5), constrained_layout=True); axs = axs.ravel()
    mesh_ax(axs[0], np.asarray(T0[0]), np.asarray(T0[1])); axs[0].set_title('(a) coarse mesh $\\mathcal{T}_0$')
    mesh_ax(axs[1], H.coords, H.elem[H.keep], lw=0.25); axs[1].set_xlim(0.1, 0.5); axs[1].set_ylim(0.12, 0.52)
    axs[1].set_title('(b) level 3, detail (3,486 unknowns)')
    t = tri_of(H)
    axs[2].tripcolor(t, facecolors=np.log10(a_ch[H.keep]), cmap='Blues', vmin=-0.5, vmax=4, edgecolors='none'); clean(axs[2])
    axs[2].set_title('(c) channels, $a=10^4$')
    fb = np.where(a_ba[H.keep] < 0.5, 0.0, 1.0); isl = islands(H, a_ba)[H.keep]; fb[isl] = 2.0
    from matplotlib.colors import ListedColormap
    axs[3].tripcolor(t, facecolors=fb, cmap=ListedColormap(['#3d3c38', '#f0efec', ORANGE]), vmin=0, vmax=2, edgecolors='none'); clean(axs[3])
    axs[3].set_title(f'(d) barriers, test sample {ib}')
    mesh_ax(axs[4], np.asarray(P0[0]), np.asarray(P0[1])); axs[4].set_title('(e) polygon $\\mathcal{T}_0$')
    te = mtri.Triangulation(He.coords[:, 0], He.coords[:, 1], He.elem[He.keep])
    axs[5].tripcolor(te, facecolors=np.log10(E[He.keep]), cmap='Blues', vmin=-0.5, vmax=2.4, edgecolors='none'); clean(axs[5])
    axs[5].set_title('(f) elasticity bands, $E_f/E_m=100$')
    fig.savefig(os.path.join(HERE, 'fig_setup.pdf')); plt.close(fig)
    print('fig_setup.pdf: barrier TEST sample', ib, 'of data seed 1 with', int(isl.sum()), 'island elements')

# ------------------------------------------------------------------ Figure: error maps
def element_energy(H, a, e_int):
    """element-wise a_e |grad e_h|^2 area_e for the P1 field with interior values e_int (zero on Dirichlet nodes)"""
    v = np.zeros(len(H.coords)); v[H.nodes[H.L]] = e_int
    X = H.coords[H.elem]; x1, y1 = X[:, 0, 0], X[:, 0, 1]; x2, y2 = X[:, 1, 0], X[:, 1, 1]; x3, y3 = X[:, 2, 0], X[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1); area = 0.5 * np.abs(det)
    u1, u2, u3 = v[H.elem[:, 0]], v[H.elem[:, 1]], v[H.elem[:, 2]]
    gx = ((u2 - u1) * (y3 - y1) - (u3 - u1) * (y2 - y1)) / det; gy = ((u3 - u1) * (x2 - x1) - (u2 - u1) * (x3 - x1)) / det
    return np.asarray(a).ravel() * (gx ** 2 + gy ** 2) * area

def case(tag, contrast, load):
    S = setup(3, 2, 1, contrast=contrast, load=load); H = S['H']; te = S['te']
    D, net, st = load_net(S, ckpath(CK[tag]), te)
    R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te]); C = apply(net, D, R, S['U_scale'])
    en = lambda A, x, u: np.sqrt(((x - u) @ (A @ (x - u))) / (u @ (A @ u)))
    e_v = np.array([en(S['A'][i], S['UC'][i] + C[j], S['U'][i]) for j, i in enumerate(te)])
    if tag == 'ch': j = int(np.argsort(e_v)[len(e_v) // 2])
    else:
        isl = np.array([islands(H, S['a'][i]).any() for i in te]); idx = np.nonzero(isl)[0]
        j = int(idx[np.argsort(e_v[idx])[len(idx) // 2]])
    i = te[j]; A = S['A'][i]; u = S['U'][i]; uc = S['UC'][i]; r = S['Fs'][i] - A @ uc
    levs, A0 = oa_hierarchy(H, A, S['nl'], 4); Z = near_kernel_basis(levs[0][0])
    x_ref = uc + classical_vcycle_proj(levs, A0, r, Z=Z, nu=4); x_vcb = uc + C[j]
    out = dict(H=H, a=S['a'][i], u=u, sid=int(i), e=dict(uc=en(A, uc, u), ref=en(A, x_ref, u), vcb=en(A, x_vcb, u)),
               dens={k: element_energy(H, S['a'][i], u - x) for k, x in (('uc', uc), ('ref', x_ref), ('vcb', x_vcb))})
    out['dens_total'] = element_energy(H, S['a'][i], u).sum()
    return out

def fig_errormaps():
    rows = [case('ch', 1e4, 'all'), case('ba', 1e-4, 'no-low')]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
    for r_, (c, nm) in enumerate(zip(rows, ('channels', 'barriers, island'))):
        H = c['H']; t = tri_of(H); v = np.zeros(len(H.coords)); v[H.nodes[H.L]] = c['u']
        m = axs[r_, 0].tripcolor(t, v, shading='gouraud', cmap='Greys'); clean(axs[r_, 0])
        axs[r_, 0].set_title('solution $u$'); axs[r_, 0].set_ylabel(f'{nm}\nsample {c["sid"]}', fontsize=8); axs[r_, 0].yaxis.set_visible(True)
        d = {k: np.maximum(x[H.keep] / c['dens_total'], 1e-12) for k, x in c['dens'].items()}
        vmax = max(x.max() for x in d.values()); norm = LogNorm(vmin=vmax * 1e-6, vmax=vmax)
        for k_, (k, lab) in enumerate((('uc', '$u_c$'), ('ref', 'reference'), ('vcb', 'OA-VCB'))):
            im = axs[r_, k_ + 1].tripcolor(t, facecolors=d[k], norm=norm, cmap='magma_r', edgecolors='none'); clean(axs[r_, k_ + 1])
            axs[r_, k_ + 1].set_title(f'{lab} ($e$ = {c["e"][k]:.3g})')
        fig.colorbar(im, ax=axs[r_, 1:], shrink=0.85, label='error energy per element / $\\|u\\|_A^2$')
    fig.savefig(os.path.join(HERE, 'fig_errormaps.pdf')); plt.close(fig)
    print('fig_errormaps.pdf:', [(c['sid'], {k: f"{v:.3e}" for k, v in c['e'].items()}) for c in rows])

# ------------------------------------------------------------------ Figure: results
def records():
    REC = []
    for f in sorted(glob.glob(os.path.join(ROOT, '**', '*.jsonl'), recursive=True)):
        rel = os.path.relpath(f, ROOT)
        if rel.split(os.sep)[0] in ('legacy', 'results', 'manuscript', 'tmp', 'feasibility', 'mirror'): continue
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if 'cfg' not in r or 'model' not in r or r.get('code') != 'f8653612eb84': continue
            res = r.get('result', r)
            if 'en_s' not in res: continue
            c = json.loads(r['cfg']) if isinstance(r['cfg'], str) else r['cfg']
            REC.append(dict(c=c, model=r['model'], en=np.array(res['en_s']), file=rel, classical=os.path.basename(f).startswith('classical')))
    return REC
def pick(REC, model, classical, **want):
    D = dict(load='all', oa_steps=4, params=22000, data_seed=0, seed=0)   # 12 Oct: seed 0 = published runs (Session 10 adds 1-2)
    hits = [r for r in REC if r['model'] == model and r['classical'] == classical and all(r['c'].get(k, D.get(k)) == v for k, v in want.items())]
    vals = {tuple(np.round(h['en'], 12)) for h in hits}
    assert len(vals) == 1, (model, want, [h['file'] for h in hits]); return hits[0]['en']
def fig_results():
    REC = records()
    CH = dict(level=3, coarse='perforated', field='channels', contrast=1e4, k0=2, load='all', oa_steps=4, params=22000, seed=0)
    BA = dict(CH, contrast=1e-4, load='no-low')
    MEM = 'V(4,4) op.-adapted + defl. + proj.'
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]; NPTS = []
    for spec, col, mk, lab in ((CH, BLUE, 'o', 'channels'), (BA, ORANGE, '^', 'barriers')):
        for s in (1, 2, 3):
            e = pick(REC, 'oa-vcb', False, **spec, data_seed=s); ref = pick(REC, MEM, True, **spec, data_seed=s)
            out = int(np.sum((np.asarray(ref) < 2e-3) | (np.asarray(ref) > 1) | (np.asarray(e) < 3e-4) | (np.asarray(e) > 1)))
            assert out == 0, f'{lab} seed {s}: {out} samples outside the plotted range'; NPTS.append(len(ref))
            ax.scatter(ref, e, s=9, marker=mk, facecolor=col, edgecolor='white', linewidth=0.3, alpha=0.85, label=lab if s == 1 else None, zorder=3)
    lim = [2e-3, 1.0]; xx = np.array(lim)                          # 11 Oct (Astra): 3e-3 clipped barrier seed 2, sample 316
    ax.plot(xx, xx, color=MUTED, lw=0.8, zorder=1); ax.plot(xx, xx / 10, color=MUTED, lw=0.8, ls='--', zorder=1)
    ax.text(0.26, 0.36, "equal error", color=MUTED, fontsize=6.5, rotation=36); ax.text(0.26, 0.0185, "10$\\times$ lower", color=MUTED, fontsize=6.5, rotation=36)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*lim); ax.set_ylim(3e-4, 1.0)
    ax.set_xlabel('Chebyshev reference, energy error'); ax.set_ylabel('OA-VCB, energy error')
    ax.set_title('(a) per test sample, seeds 1--3'); ax.legend(frameon=False, loc='upper left')
    ax.grid(True, which='major', color='#e6e5e1', lw=0.5, zorder=0)
    ax = axs[1]; LBL = []
    for spec, col, mk, lab in ((CH, BLUE, 'o', 'channels'), (BA, ORANGE, '^', 'barriers')):
        H = np.array([pick(REC, f'two-grid PCG, {k} its', True, **spec, data_seed=1) for k in range(1, 17)])
        q1, md, q3 = np.percentile(H, [25, 50, 75], axis=1); k = np.arange(1, 17)
        ax.fill_between(k, q1, q3, color=col, alpha=0.15, lw=0); ax.plot(k, md, color=col, lw=2, label=f'two-grid PCG, {lab}')
        e = pick(REC, 'oa-vcb', False, **spec, data_seed=1); ref = pick(REC, MEM, True, **spec, data_seed=1)
        ax.axhline(np.median(e), color=col, lw=1.2, ls='-', alpha=0.9); ax.axhline(np.median(ref), color=col, lw=1.2, ls=':', alpha=0.9)
        LBL.append((np.median(e), f'OA-VCB, {lab}', col)); LBL.append((np.median(ref), f'reference, {lab}', col))
    LBL.sort(); ys = [np.log10(y) for y, _, _ in LBL]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < 0.62: ys[i] = ys[i - 1] + 0.62
    for (y, t_, col), yl in zip(LBL, ys):
        ax.annotate(t_, xy=(16, y), xytext=(16.4, 10 ** yl), fontsize=6.5, color=INK, va='center', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.6))
    ax.set_yscale('log'); ax.set_xlim(1, 16); ax.set_xlabel('two-grid PCG iterations (from $u_c$)'); ax.set_ylabel('energy error (median, quartiles)')
    from matplotlib.lines import Line2D
    hs, ls_ = ax.get_legend_handles_labels()
    hs += [Line2D([], [], color=MUTED, lw=1.2), Line2D([], [], color=MUTED, lw=1.2, ls=':')]; ls_ += ['OA-VCB, one pass (median)', 'reference, one pass (median)']
    ax.set_title('(b) one pass vs PCG iterations, seed 1'); ax.legend(hs, ls_, frameon=False, loc='lower left')
    ax.grid(True, which='major', color='#e6e5e1', lw=0.5)
    fig.savefig(os.path.join(HERE, 'fig_results.pdf')); plt.close(fig)
    print('fig_results.pdf written;', sum(NPTS), 'samples plotted, none outside the axes')

# ------------------------------------------------------------------ Figure: accuracy versus time (12 Oct)
def fig_avt():
    """Mean relative energy error against mean time per sample, one CPU, from diagnostics_cost/accuracy_vs_time*.json
    (accuracy_vs_time.py; channels and barriers, held-out seed 1).  Colours: OA-VCB blue, classical one-pass family orange,
    PCG preconditioners aqua with distinct markers / line styles (palette checked with the dataviz validator, 3 slots, all
    pairs; aqua is below 3:1 contrast, so every series is also identified by marker and legend)."""
    import json
    from matplotlib.lines import Line2D
    DCD = os.path.join(ROOT, 'diagnostics_cost')
    J = {'channels': json.load(open(os.path.join(DCD, 'accuracy_vs_time.json'))),
         'barriers': json.load(open(os.path.join(DCD, 'accuracy_vs_time_barriers.json')))}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.1), constrained_layout=True)   # 13 Oct: taller, for a larger legend
    PCGS = (('two-grid PCG', 'o', '-', 'two-grid PCG, 1, 2, ... iterations'), ('SA-AMG PCG', 's', '--', 'SA-AMG PCG, 1, 2, ... iterations'),
            ('PCG + classical V(4,4)', '^', ':', 'PCG with classical V(4,4), 1, 2, ...'))
    for ax, (fam, Jf) in zip(axs, J.items()):
        M = Jf['methods']
        for name, mk, ls, _ in PCGS:
            c = M[name]['curve']
            ax.plot([q['ms_mean'] for q in c], [q['energy_mean'] for q in c], ls=ls, marker=mk, ms=3.2, lw=1.2, color=AQUA, mec='white', mew=0.4, zorder=3)
        one = [M[f'reference, Chebyshev nu={n}' + (' (paper)' if n == 4 else '')] for n in (4, 8, 16)]
        ax.plot([m['ms_mean'] for m in one], [m['energy_mean'] for m in one], ls='-', lw=1.0, marker='D', ms=3.6, color=ORANGE, mec='white', mew=0.4, zorder=4)
        for n, m, dy in ((4, one[0], 3), (16, one[2], -3)):
            ax.annotate(f'$\\nu={n}$', xy=(m['ms_mean'], m['energy_mean']), xytext=(-6, dy), textcoords='offset points', ha='right',
                        va='bottom' if dy > 0 else 'top', fontsize=6.5, color=INK)
        rep = [M['reference, Chebyshev nu=4 (paper)']] + [M[f'reference nu=4, applied {k} times'] for k in (2, 3, 4)]
        ax.plot([m['ms_mean'] for m in rep], [m['energy_mean'] for m in rep], ls='--', lw=1.0, marker='o', ms=3.2, color=ORANGE, mfc='white', mew=0.9, zorder=4)
        g = M['reference, symmetric Gauss-Seidel 4 sweeps']
        ax.plot([g['ms_mean']], [g['energy_mean']], marker='v', ms=4, color=ORANGE, mfc='white', mew=0.9, ls='none', zorder=4)
        v = M['OA-VCB (one pass)']
        ax.plot([v['ms_mean']], [v['energy_mean']], marker='*', ms=9, color=BLUE, mec='white', mew=0.5, ls='none', zorder=5)
        ax.annotate('OA-VCB', xy=(v['ms_mean'], v['energy_mean']), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7, color=INK, fontweight='bold')
        ax.axvline(M['sparse direct (SuperLU)']['ms_mean'], color=MUTED, lw=0.8, ls='-.', zorder=1)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(4, 450)
        ax.set_xlabel('time per sample (ms, one CPU)'); ax.set_ylabel('arithmetic mean relative energy error')
        ax.set_title(f"({'a' if fam == 'channels' else 'b'}) {fam}, held-out seed 1")
        ax.grid(True, which='major', color='#e6e5e1', lw=0.5); ax.set_axisbelow(True)
        for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    H_ = [Line2D([], [], marker='*', ms=8, color=BLUE, ls='none', mec='white', label='OA-VCB, one pass'),
          Line2D([], [], marker='D', ms=3.6, color=ORANGE, ls='-', lw=1.0, mec='white', label='classical ref., one pass, Chebyshev $\\nu=4, 8, 16$'),
          Line2D([], [], marker='v', ms=4, color=ORANGE, mfc='white', ls='none', label='classical ref., one pass, symm. Gauss–Seidel'),
          Line2D([], [], marker='o', ms=3.2, color=ORANGE, mfc='white', ls='--', lw=1.0, label='classical ref. ($\\nu=4$) applied 1–4 times')]
    H_ += [Line2D([], [], marker=mk, ms=3.2, color=AQUA, ls=ls, lw=1.2, mec='white', label=lab) for _, mk, ls, lab in PCGS]
    H_ += [Line2D([], [], color=MUTED, ls='-.', lw=0.8, label='sparse direct solve (time only; error below range)')]
    fig.legend(handles=H_, loc='outside lower center', ncol=2, frameon=False, fontsize=7.5, handlelength=3.6)
    fig.savefig(os.path.join(HERE, 'fig_avt.pdf')); plt.close(fig)
    print('fig_avt.pdf written; platform', J['channels']['cpu'], J['channels']['threads'], 'threads')

if __name__ == '__main__':
    which = sys.argv[1:] or ['setup', 'errormaps', 'results', 'avt']
    torch.set_num_threads(2)
    for w in which: {'setup': fig_setup, 'errormaps': fig_errormaps, 'results': fig_results, 'avt': fig_avt}[w]()

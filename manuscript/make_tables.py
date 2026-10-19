"""Tables of Sections 4-5 and the appendix, generated from the raw result records (no number typed by hand).
Run from the OFW_NO folder:   python manuscript/make_tables.py [root]      (root defaults to OFW_NO)
Writes manuscript/tables/*.tex, manuscript/tables/numbers.txt and manuscript/tables/MANIFEST.txt (every input file used).
Version 2 (11 Oct, after SECTIONS_4_5_REVIEW_ASTRA): ambiguous or missing records and sample-id mismatches raise; no
silent skips; the island split cannot fall back to a stale table; censored PCG histories are never filled.
Version 4 (13 Oct, after FINAL_SUBMISSION_REVIEW_ASTRA): compact main-text table tab_core (same records as tab_channels /
tab_barriers, which move to the appendix); tab_initseeds in two panels; short labels in tab_published; captions of tab_avt and
tab_excluded corrected (metric and containment wording).  No number changes.
Version 3 (11 Oct, after SECTIONS_4_5_FOLLOWUP_ASTRA): sample ids are required (stored ids, or the driver split from the
record's n_train / n_test); conflicting duplicate records raise in every lookup, classical and settings tables included.
Statistics: e = per-sample relative energy error ||u - x||_A / ||u||_A on test ids 300-399 of each data seed; paired
G = exp(mean(log(e_ref / e_model))) (95% CI: 5,000 paired bootstrap resamples, rng 0); W = fraction with e_model < e_ref;
"s.d." = sample standard deviation (ddof 1) of the per-seed means (descriptive, three seeds; not an initialisation
uncertainty).  Island label for the barrier split: geometric (conductive component touching no Dirichlet node)."""
import json, glob, os, sys, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
OUT = os.path.join(HERE, 'tables'); os.makedirs(OUT, exist_ok=True)
DC = os.path.join(ROOT, 'diagnostics_cost')
CODE = 'f8653612eb84'; MEM = 'V(4,4) op.-adapted + defl. + proj.'; IDS = list(range(300, 400))
NUM = open(os.path.join(OUT, 'numbers.txt'), 'w'); USED = set()
def note(s): print(s); NUM.write(s + '\n')

# ------------------------------------------------------------------ records
def rec_ids(r, res):
    """test sample ids of a record: stored ids, else the driver's split (test = n_train .. n_train + n_test - 1) from the
    record's own n_train / n_test; None if neither is present (such a record is rejected when used)."""
    ids = r.get('sample_ids', res.get('sample_ids'))
    if ids is not None: return [int(i) for i in ids]
    if 'n_train' in r and 'n_test' in r: return list(range(int(r['n_train']), int(r['n_train']) + int(r['n_test'])))
    return None

def check_ids(r, what):
    if r['ids'] is None: raise ValueError(f"{what}: record {r['file']} has neither sample ids nor n_train / n_test")
    if r['ids'] != IDS: raise ValueError(f"{what}: unexpected sample ids in {r['file']}")

REC = []
for f in sorted(glob.glob(os.path.join(ROOT, '**', '*.jsonl'), recursive=True)):
    rel = os.path.relpath(f, ROOT)
    if rel.split(os.sep)[0] in ('legacy', 'results', 'manuscript', 'tmp', 'feasibility'): continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if 'cfg' not in r or 'model' not in r: continue
        res = r.get('result', r)
        if 'en_s' not in res or r.get('code') != CODE: continue
        c = json.loads(r['cfg']) if isinstance(r['cfg'], str) else r['cfg']
        REC.append(dict(c=c, model=r['model'], en=np.array(res['en_s'], float),
                        rl2=np.array(res.get('rl2_s', [np.nan] * len(res['en_s'])), float), res=res, file=rel,
                        ids=rec_ids(r, res), classical=os.path.basename(f).startswith(('classical', 'random_classical', 'linear_controls_classical'))))
note(f"{len(REC)} records of code version {CODE} under {ROOT}")
# 12 Oct (REVIEW_AND_PURPOSE_ASTRA, item 3): the initialisation seed is part of every standard lookup.  The published
# tables use seed 0; records without a 'seed' key are seed 0.  Session 10 adds seeds 1-2, which are aggregated separately
# (section "initialisation seeds" at the end) and can never make a standard lookup ambiguous.
DFLT = dict(load='all', oa_steps=4, data_seed=0, params=22000, seed=0)

def match(r, want):
    return all(r['c'].get(k, DFLT.get(k)) == v for k, v in want.items())

def get(model, prefer=None, classical=False, **want):
    want = dict(dict(load='all', oa_steps=4, params=22000, seed=0), **want)
    hits = [r for r in REC if r['model'] == model and r['classical'] == classical and match(r, want)]
    if prefer is not None: hits = [h for h in hits if prefer in h['file']] or hits
    distinct = {tuple(np.round(h['en'], 12)) for h in hits}
    if not hits: raise KeyError(f"no record: {model} {want}")
    if len(distinct) > 1: raise ValueError(f"ambiguous records for {model} {want}: {[h['file'] for h in hits]}")
    r = hits[0]; USED.add(r['file'])
    if len(r['en']) != 100: raise ValueError(f"{model} {want}: {len(r['en'])} samples")
    check_ids(r, f"{model} {want}")
    return r

def G(ref, model):
    lr = np.log(ref / model); b = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr)))
    return np.exp(lr.mean()), np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5])), float(np.mean(lr > 0))

def pcg_its(hist, e, nmax):
    """hist: (n, nmax) energy errors after 1..nmax iterations; returns median, min, max, number censored (> nmax)."""
    hit = [int(np.nonzero(h <= t)[0][0]) + 1 if (h <= t).any() else None for h, t in zip(hist, e)]
    cen = sum(h is None for h in hit); vals = [h if h is not None else nmax + 1 for h in hit]
    return np.median(vals), min(vals), (max(vals) if cen == 0 else f'>{nmax}'), cen

def fmt(x): return f"{x:.2e}".replace('e-0', 'e-').replace('e+0', 'e')
def gtx(g): return f"{g[0]:.2f} [{g[1][0]:.2f}, {g[1][1]:.2f}]"
def write(name, caption, label, colspec, header, lines, size='\\footnotesize', resize=False):
    tab = f"\\begin{{tabular}}{{{colspec}}}\\toprule\n{header} \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}"
    tab = "\\begin{adjustbox}{max width=\\linewidth}\n" + tab + "\n\\end{adjustbox}"     # shrink only if wider than the text
    open(os.path.join(OUT, f'{name}.tex'), 'w').write(f"\\begin{{table}}[tbp]\\centering{size}\\setlength{{\\tabcolsep}}{{4pt}}\n\\caption{{{caption}}}\\label{{{label}}}\n{tab}\n\\end{{table}}\n")

def write_panels(name, caption, label, panels, size='\\footnotesize'):
    """table with stacked panels; panels = [(title, colspec, header, lines), ...]; each panel shrinks only if too wide."""
    body = []
    for k, (title, colspec, header, lines) in enumerate(panels):
        tab = f"\\begin{{tabular}}{{{colspec}}}\\toprule\n{header} \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}"
        body.append(("\\par\\medskip\n" if k else "") + f"{{\\small {title}}}\\par\\smallskip\n\\begin{{adjustbox}}{{max width=\\linewidth}}\n" + tab + "\n\\end{adjustbox}")
    open(os.path.join(OUT, f'{name}.tex'), 'w').write(f"\\begin{{table}}[tbp]\\centering{size}\\setlength{{\\tabcolsep}}{{4pt}}\n\\caption{{{caption}}}\\label{{{label}}}\n" + "\n".join(body) + "\n\\end{table}\n")

CH = dict(level=3, coarse='perforated', field='channels', contrast=1e4, k0=2)
BA = dict(level=3, coarse='perforated', field='channels', contrast=1e-4, k0=2, load='no-low')
SEEDS = (1, 2, 3)
def classical(spec, s, oa_steps=4):
    out = {}
    for r in REC:
        if not (r['classical'] and match(r, dict(dict(load='all', oa_steps=oa_steps, params=22000, seed=0), **spec, data_seed=s))): continue
        if r['model'] in out and not np.array_equal(np.round(out[r['model']]['en'], 12), np.round(r['en'], 12)):
            raise ValueError(f"classical {spec} seed {s}: conflicting records for {r['model']}: {out[r['model']]['file']}, {r['file']}")
        out.setdefault(r['model'], r)
    for r in out.values():
        USED.add(r['file']); check_ids(r, f"classical {spec} seed {s}")
        if len(r['en']) != 100: raise ValueError(f"classical {spec} seed {s} {r['model']}: {len(r['en'])} samples")
    if MEM not in out: raise KeyError(f"no classical member for {spec} seed {s}")
    return {k: r['en'] for k, r in out.items()}, {k: r['rl2'] for k, r in out.items()}

# ------------------------------------------------------------------ Tables 1-2 (aggregate) and the per-seed panel
ROWS = [('oa-vcb', 'OA-VCB (ours)'), ('oa-vc', 'OA-VC'), ('oa-ll-op', 'OA-Lift'), ('gnn-op', 'GNN-op'), ('gnn', 'GNN'),
        ('transolver-op', 'Transolver-op'), ('fno-op', 'FNO-op$^\\dagger$'), ('wno-op', 'WNO-op$^\\dagger$'),
        ('lino-op', 'LiNO-op$^\\dagger$'), ('mgno', 'MgNO$^\\dagger$'), ('mgno-op', 'MgNO-op$^\\dagger$')]
CLROWS = [(MEM, 'classical reference'), ('V(4,4) operator-adapted', 'classical V(4,4), no defl./proj.'),
          ('two-grid PCG, 4 its', 'two-grid PCG, 4 its'), ('two-grid PCG, 6 its', 'two-grid PCG, 6 its'),
          ('two-grid PCG, 8 its', 'two-grid PCG, 8 its'), ('coarse solve alone', '$u_c$ alone')]
PERSEED = {}; CORE = {}
def class_table(spec, name, caption, label, prefer):
    cl = {s: classical(spec, s) for s in SEEDS}; mem = np.concatenate([cl[s][0][MEM] for s in SEEDS]); lines = []; full = []
    def line(lab, per, rl2, params='--', is_mem=False):
        means = [x.mean() for x in per]; e = np.concatenate(per); g = G(mem, e); CORE[(name, lab)] = (np.mean(means), g, is_mem)
        lines.append(f"{lab} & {params} & {fmt(np.mean(means))} $\\pm$ {fmt(np.std(means, ddof=1))} & {fmt(np.mean([x.mean() for x in rl2]))} & "
                     + ("1 & --" if is_mem else f"{gtx(g)} & {g[2]:.2f}") + " \\\\")
        return means
    for key, lab in ROWS:
        recs = [get(key, prefer=prefer.get(key), **spec, data_seed=s) for s in SEEDS]
        p = recs[0]['res'].get('params'); means = line(lab, [r['en'] for r in recs], [r['rl2'] for r in recs], f"{p:,}".replace(',', '{,}'))
        nw = [r['res'].get('n_worse_than_uc') for r in recs]
        full.append(f"{lab} & " + " & ".join(fmt(m) for m in means) + " & " + " / ".join(str(x) for x in nw) + " \\\\")
        if key in ('oa-vcb', 'oa-vc'): PERSEED[(name, key)] = [r['en'] for r in recs]
    lines.append('\\midrule')
    for key, lab in CLROWS:
        line(lab, [cl[s][0][key] for s in SEEDS], [cl[s][1][key] for s in SEEDS], is_mem=(key == MEM))
    PERSEED[(name, 'member')] = [cl[s][0][MEM] for s in SEEDS]
    write(name, caption, label, 'lrllll', 'model & params & energy (mean $\\pm$ s.d.) & nodal rel.\\ $\\ell^2$ & $G$ [95\\% CI] & $W$', lines)
    write(name + '_full', f"Per-seed mean energy errors and the number of test samples on which the model is worse in energy than $u_c$ (seeds 1 / 2 / 3); {caption.split('.')[0].lower()}.",
          label + 'full', 'lllll', 'model & seed 1 & seed 2 & seed 3 & worse than $u_c$', full)
    for s in SEEDS:
        e = get('oa-vcb', **spec, data_seed=s)['en']; H = np.array([cl[s][0][f'two-grid PCG, {j} its'] for j in range(1, 17)]).T
        m, a, b, c = pcg_its(H, e, 16); g = G(cl[s][0][MEM], e)
        note(f"[{name}] seed {s}: OA-VCB {e.mean():.4e} member {cl[s][0][MEM].mean():.4e} G {gtx(g)} W {g[2]:.2f} worst {np.min(cl[s][0][MEM] / e):.2f}; PCG its median {m:.0f} ({a}-{b})")
    return cl
note('== Table 1: channels'); CL1 = class_table(CH, 'tab_channels',
    "Channels (contrast $10^4$, L3, $k_0=2$), held-out data seeds 1--3, 100 test operators each. Energy error: mean and sample s.d.\\ of the three per-seed means; "
    "$G$: paired geometric-mean ratio of the classical reference's error to the model's over the pooled 300 samples; $W$: fraction of samples on which the model is more "
    "accurate than the reference. $^\\dagger$ Grid models, through the grid bridge (Section~\\ref{sec:baselines}).", 'tab:channels', prefer={'wno-op': '3-3b'})
note('== Table 2: barriers'); CL2 = class_table(BA, 'tab_barriers',
    "Barriers (contrast $10^{-4}$, source only where $a\\ge1/2$, L3, $k_0=2$), held-out data seeds 1--3. Columns as in Table~\\ref{tab:channels}. "
    "The barrier WNO-op runs used the fallback discrete wavelet transform (\\appref{app:baselines}).", 'tab:barriers', prefer={})
# 13 Oct (FINAL_SUBMISSION_REVIEW_ASTRA, section 3): compact main-text comparison; the full baseline tables move to the appendix.
CORE_ROWS = [('OA-VCB (ours)', 'OA-VCB (ours)'), ('OA-VC', 'OA-VC (no deflation, plain output)'), ('OA-Lift', 'OA-Lift'), ('GNN-op', 'GNN-op'),
             ('Transolver-op', 'Transolver-op'), (None, None), ('classical reference', 'classical reference (Chebyshev, $\\nu=4$)'),
             ('two-grid PCG, 6 its', 'two-grid PCG, 6 iterations'), ('$u_c$ alone', '$u_c$ alone')]
lines = []
for key, lab in CORE_ROWS:
    if key is None: lines.append('\\midrule'); continue
    row = [lab]
    for name in ('tab_channels', 'tab_barriers'):
        m, g, is_mem = CORE[(name, key)]
        row += [fmt(m)] + (['1', '--'] if is_mem else [gtx(g), f"{g[2]:.2f}"])
    lines.append(' & '.join(row) + ' \\\\')
write('tab_core', "Main comparison, held-out data seeds 1--3 (100 test operators each; L3, $k_0=2$). Energy: mean of the three per-seed "
      "arithmetic mean relative energy errors. $G$: paired geometric-mean ratio of the classical reference's error to the model's over the pooled 300 "
      "samples [95\\% CI]; $W$: fraction of samples on which the model is more accurate than the reference. All learned rows have 19{,}521--21{,}603 "
      "parameters. The other learned baselines (GNN, and the grid-based FNO-op, WNO-op, LiNO-op, MgNO, MgNO-op, which enter through a mesh--grid bridge) "
      "and further classical rows are in Tables~\\ref{tab:channels} and~\\ref{tab:barriers}.", 'tab:core', 'lllllll',
      ' & \\multicolumn{3}{c}{channels (contrast $10^4$)} & \\multicolumn{3}{c}{barriers (contrast $10^{-4}$)} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n'
      'model & energy & $G$ [95\\% CI] & $W$ & energy & $G$ [95\\% CI] & $W$', lines)
lines = []
for name, lab, cl in (('tab_channels', 'channels', CL1), ('tab_barriers', 'barriers', CL2)):
    for i, s in enumerate(SEEDS):
        e = PERSEED[(name, 'oa-vcb')][i]; m = PERSEED[(name, 'member')][i]; v = PERSEED[(name, 'oa-vc')][i]
        g = G(m, e); gv = G(m, v)
        lines.append(f"{lab} & {s} & {fmt(e.mean())} & {fmt(m.mean())} & {gtx(g)} & {g[2]:.2f} & {fmt(v.mean())} & {gv[0]:.2f}, {gv[2]:.2f} \\\\")
    if lab == 'channels': lines.append('\\midrule')
write('tab_perseed', "Per-seed results for OA-VCB, its classical reference and OA-VC (mean energy errors; reference/model $G$ [95\\% CI] and $W$).", 'tab:perseed',
      'llllllll', 'class & seed & OA-VCB & reference & $G$ [95\\% CI] & $W$ & OA-VC & $G$, $W$', lines, resize=True)

# ------------------------------------------------------------------ islands (geometric label)
sys.path.insert(0, ROOT); sys.path.insert(0, DC)
from s1_common import setup
from harmonic_study import geometric_island
lines = []
for s in SEEDS:
    S = setup(3, 2, s, contrast=1e-4, load='no-low'); isl = np.array([geometric_island(S['H'], S['a'][i]) for i in S['te']])
    e = get('oa-vcb', **BA, data_seed=s)['en']; m = CL2[s][0][MEM]; c = CL2[s][0]['coarse solve alone']
    gi, gn = G(m[isl], e[isl]), G(m[~isl], e[~isl])
    lines.append(f"{s} & {isl.sum()} & {e[isl].mean():.3f} / {m[isl].mean():.3f} & {gi[0]:.2f}, {gi[2]:.2f} & {e[~isl].mean():.4f} / {m[~isl].mean():.4f} & {gn[0]:.2f}, {gn[2]:.2f} & {e[isl].mean() / e[~isl].mean():.1f} \\\\")
    note(f"[islands] seed {s}: {isl.sum()} island samples; island {e[isl].mean():.3f}/{m[isl].mean():.3f} G {gi[0]:.2f} W {gi[2]:.2f}; no island {e[~isl].mean():.4f}/{m[~isl].mean():.4f} G {gn[0]:.2f} W {gn[2]:.2f}; "
         f"OA-VCB island/non-island {e[isl].mean() / e[~isl].mean():.1f}, member {m[isl].mean() / m[~isl].mean():.1f}; worse than u_c {(e > c * (1 + 1e-6)).sum()}")
write('tab_islands', "Barriers split by floating islands (geometric label, conductive component touching no Dirichlet node). Mean energy error OA-VCB / classical reference, "
      "paired $G$ and $W$ within each group, and OA-VCB's ratio of island to non-island mean error.", 'tab:islands', 'rrlllll',
      'seed & islands & with island & $G$, $W$ & without island & $G$, $W$ & ratio', lines)

# ------------------------------------------------------------------ Table 3: factorial (+ eight-cell output/loss panel)
note('== Table 3: factorial')
FAC = [('oa-vc', 'OA-VC', 'no', 'plain', 'plain', 'learned'), ('oa-vcz', 'OA-VCZ', 'yes', 'plain', 'plain', 'learned'),
       ('oa-vcq', 'OA-VCQ', 'no', 'Galerkin', 'plain + proj.', 'learned'), ('oa-vcb', 'OA-VCB', 'yes', 'Galerkin', 'plain + proj.', 'learned'),
       ('oa-vcl', 'OA-VCL', 'yes', 'plain', 'plain', 'residual-linear'), ('oa-vcbl', 'OA-VCBL', 'yes', 'Galerkin', 'plain + proj.', 'residual-linear')]
lines = []; cells8 = {}
for key, lab, dfl, out, loss, sm in FAC:
    row = [lab, dfl, out, loss, sm]
    for nm, spec, cl in (('barriers', BA, CL2[1][0]), ('channels', CH, CL1[1][0])):
        try: r = get(key, **spec, data_seed=1, prefer='linear_controls' if (nm == 'channels' and key in ('oa-vcz', 'oa-vcl', 'oa-vcbl')) else None)
        except KeyError:
            row += ['--', '--']; continue
        g = G(cl[MEM], r['en']); row += [fmt(r['en'].mean()), f"{g[0]:.2f}, {g[2]:.2f}"]
        if nm == 'barriers': row_params = r['res'].get('params')
        res = r['res']; plain = res['energy'] if 'energy_proj' in res else res.get('energy_plain'); proj = res.get('energy_proj', res['energy'])
        cells8[(nm, key)] = (plain, proj)
        note(f"  {lab} {nm}: {r['en'].mean():.4e} member/model G {gtx(g)} W {g[2]:.2f}; plain {plain:.4f} proj {proj:.4f} ({r['file']})")
    lines.append(" & ".join(row[:5] + [f"{row_params:,}".replace(',', '{,}')] + row[5:]) + " \\\\")
for (a, b) in (('oa-vc', 'oa-vcz'), ('oa-vcq', 'oa-vcb'), ('oa-vc', 'oa-vcq'), ('oa-vcz', 'oa-vcb'), ('oa-vcl', 'oa-vcz'), ('oa-vcbl', 'oa-vcb')):
    for nm, spec in (('barriers', BA), ('channels', CH)):
        try:
            ra = get(a, **spec, data_seed=1, prefer='linear_controls' if (nm == 'channels' and a in ('oa-vcz', 'oa-vcl', 'oa-vcbl')) else None)
            rb = get(b, **spec, data_seed=1, prefer='linear_controls' if (nm == 'channels' and b in ('oa-vcz', 'oa-vcl', 'oa-vcbl')) else None)
        except KeyError: continue
        g = G(ra['en'], rb['en']); note(f"  factorial {nm}: {a}/{b} G {gtx(g)} W {g[2]:.2f}")
write('tab_factorial', "Separately trained design variants, held-out data seed 1. Deflation, output used at inference (plain cycle or Galerkin projection), training loss and smoother class; "
      "mean energy error with each model's own output and reference/model $G$, $W$. OA-VCQ was not run on channels, where no near-kernel vectors are detected.", 'tab:factorial',
      'llllll' + 'llll', ' & deflation & output & loss & smoothers & params & \\multicolumn{2}{c}{barriers: energy, ($G$, $W$)} & \\multicolumn{2}{c}{channels: energy, ($G$, $W$)}', lines, resize=True)
lines = []
for key, lab, dfl, out, loss, sm in FAC[:4] + FAC[4:]:
    for nm in ('barriers', 'channels'):
        if (nm, key) in cells8:
            p, q = cells8[(nm, key)]; lines.append(f"{nm} & {lab} & {dfl} & {loss} & {sm} & {p:.4f} & {q:.4f} \\\\")
write('tab_factorial8', "Plain and projected outputs of the same trained weights (mean energy error, held-out seed 1). Models with loss ``plain'' optimise the plain output only; "
      "models with loss ``plain + proj.'' optimise both output losses, and their projected output is the one deployed. In each row the other column is the same network evaluated with the other output.", 'tab:factorial8', 'lllllll',
      'class & model & deflation & loss & smoothers & plain output & projected output', lines)

# ------------------------------------------------------------------ Table 4: further settings
note('== Table 4: further settings'); lines = []
def setting(lab, e, m, hist=None, nmax=16):
    g = G(m, e); its = pcg_its(hist, e, nmax) if hist is not None else None
    lines.append(f"{lab} & {fmt(e.mean())} & {fmt(m.mean())} & {gtx(g)} & {g[2]:.2f} & " + (f"{its[0]:.0f} ({its[1]}--{its[2]})" if its else '--') + " \\\\")
    note(f"  {lab}: OA-VCB {e.mean():.4e} member {m.mean():.4e} G {gtx(g)} W {g[2]:.2f}" + (f" PCG its {its}" if its else ''))
rvl = [json.loads(l) for l in open(os.path.join(DC, 'k0_0_oa-vcb_reeval.jsonl'))]
rv = {json.loads(v['cfg'])['data_seed']: v for v in rvl}
if len(rv) != len(rvl): raise ValueError("k0_0_oa-vcb_reeval.jsonl: duplicate seeds")
USED.add('diagnostics_cost/k0_0_oa-vcb_reeval.jsonl')
for s in SEEDS:
    assert rv[s]['sample_ids'] == IDS
    k0l = [json.loads(l) for l in open(os.path.join(DC, f'k0_0_classical_seed{s}.jsonl'))]; USED.add(f'diagnostics_cost/k0_0_classical_seed{s}.jsonl')
    if len({v['model'] for v in k0l}) != len(k0l): raise ValueError(f"k0_0_classical_seed{s}.jsonl: duplicate model records")
    k0c = {v['model']: v for v in k0l}
    assert all(v['sample_ids'] == IDS for v in k0c.values())
    hist = np.array([k0c[f'two-grid PCG, {j} its, from 0']['en_s'] for j in range(1, 13)]).T
    setting(f"$k_0=0$ (no external coarse solution), seed {s}", np.array(rv[s]['result']['en_s']), np.array(k0c['V(4,4) op.-adapted + defl. + proj., from 0']['en_s']), hist, 12)
def by_model(recs, what):
    """one record per model name: conflicting duplicates raise; ids and length are checked; inputs recorded."""
    out = {}
    for r in recs:
        if r['model'] in out and not np.array_equal(np.round(out[r['model']]['en'], 12), np.round(r['en'], 12)):
            raise ValueError(f"{what}: conflicting records for {r['model']}: {out[r['model']]['file']}, {r['file']}")
        out.setdefault(r['model'], r)
    for r in out.values():
        USED.add(r['file']); check_ids(r, what)
        if len(r['en']) != 100: raise ValueError(f"{what} {r['model']}: {len(r['en'])} samples")
    return {k: r['en'] for k, r in out.items()}
def cls_of(sub, what='classical'):
    return by_model([r for r in REC if r['classical'] and sub(r)], what)
c4 = cls_of(lambda r: r['c'].get('level') == 4); h4 = np.array([c4[f'two-grid PCG, {j} its'] for j in range(1, 17)]).T
setting("finer hierarchy (L4, $k_0=3$), seed 1", get('oa-vcb', level=4, coarse='perforated', field='channels', contrast=1e4, k0=3, data_seed=1)['en'], c4[MEM], h4)
cp = cls_of(lambda r: r['c'].get('coarse') == 'polygon'); hp = np.array([cp[f'two-grid PCG, {j} its'] for j in range(1, 17)]).T
setting("polygonal domain, seed 1", get('oa-vcb', level=3, coarse='polygon', field='channels', contrast=1e4, k0=2, data_seed=1)['en'], cp[MEM], hp)
for s in SEEDS:
    cr = cls_of(lambda r, s=s: r['c'].get('load') == 'random' and r['c'].get('data_seed') == s); hr = np.array([cr[f'two-grid PCG, {j} its'] for j in range(1, 17)]).T
    setting(f"random loads (in distribution), seed {s}", get('oa-vcb', **dict(CH, load='random'), data_seed=s)['en'], cr[MEM], hr)
for s in (0, 1, 2):
    A = by_model([r for r in REC if r['c'].get('pde') == 'aniso' and r['c'].get('data_seed', 0) == s], f"aniso seed {s}")
    setting(f"anisotropic tensor coefficient, seed {s}" + (' (dev.)' if s == 0 else ''), A['oa-vcb'], A['classical member (+ defl. + proj.)'])
write('tab_settings', "Further settings: mean energy error of OA-VCB and of its classical reference, paired reference/OA-VCB $G$ [95\\% CI] and $W$, and the per-sample number of two-grid "
      "PCG iterations, from the same starting point, at which PCG first reaches OA-VCB's energy error (median, range; computed iteration histories: 12 for $k_0=0$, 16 otherwise).",
      'tab:settings', 'llllll', 'setting & OA-VCB & reference & $G$ [95\\% CI] & $W$ & PCG its', lines)

# ------------------------------------------------------------------ Table 5: transfers
note('== Table 5: transfers'); lines = []
for lab, spec, s, CL in (('channels', CH, 1, CL1), ('channels', CH, 2, CL1), ('channels', CH, 3, CL1), ('barriers', BA, 1, CL2)):
    ad = get('oa-vcb', **spec, data_seed=s); ge = get('oa-vcb', **spec, data_seed=s, oa_steps=0)
    cg = classical(spec, s, oa_steps=0)[0]; ca = CL[s][0]
    g = G(ge['en'], ad['en']); la = G(ca[MEM], ad['en']); lg = G(cg[MEM], ge['en']); gm = G(cg[MEM], ca[MEM])
    lines.append(f"{lab}, seed {s} & {fmt(ad['en'].mean())} & {fmt(ge['en'].mean())} & {gtx(g)} & {g[2]:.2f} & {gm[0]:.2f} & {la[0]:.2f} & {lg[0]:.2f} \\\\")
    note(f"  {lab} s{s}: adapted {ad['en'].mean():.3e} geometric {ge['en'].mean():.3e} G {gtx(g)} W {g[2]:.2f}; member geo/adapted {gm[0]:.2f}; member/learned adapted {la[0]:.2f} geometric {lg[0]:.2f}")
write('tab_transfers', "Operator-adapted ($m=4$) vs geometric ($m=0$) transfers, separately trained OA-VCB models on the same data. OA-VCB error with each; paired $G$ "
      "(geometric-transfer error / adapted-transfer error) and $W$; the same ratio for the two classical references; and reference/learned $G$ for each choice of transfers.", 'tab:transfers',
      'llllllll', 'case & adapted & geometric & $G$ geom./adapt. & $W$ & references: geom./adapt. & reference/learned (adapt.) & reference/learned (geom.)', lines, resize=True)

# ------------------------------------------------------------------ Table 7: harmonic enrichment
note('== Table 7: harmonic enrichment')
def hstats(R, who, rho=2):
    isl = [r for r in R if r['n_islands'] > 0]; n = len(isl)
    e0 = np.array([r['sets'][who]['E0']['err'] for r in isl]); e2 = np.array([r['sets'][who][f'E2_{rho}']['err'] for r in isl])
    e4 = np.array([r['sets'][who][f'E4_{rho}']['err'] for r in isl]); kap = np.array([r['sets'][who][f'E2_{rho}']['kappa'] for r in isl])
    b = np.random.default_rng(0).integers(0, n, (5000, n))
    Rc = 1 - e2.mean() / e0.mean(); Rci = np.percentile(1 - e2[b].mean(1) / e0[b].mean(1), [2.5, 97.5])
    lg = np.log(e4 / e2); g24 = np.exp(lg.mean()); gci = np.exp(np.percentile(lg[b].mean(1), [2.5, 97.5]))
    sub = [r for r in isl if 'E3' in r['sets'][who]]
    s0 = np.mean([r['sets'][who]['E0']['err'] for r in sub]); sl = 1 - np.mean([r['sets'][who][f'E2_{rho}']['err'] for r in sub]) / s0
    sg = 1 - np.mean([r['sets'][who]['E3']['err'] for r in sub]) / s0
    return dict(N=len(R), n=n, e0=e0.mean(), e2=e2.mean(), R=Rc, Rci=Rci, g24=g24, gci=gci, kq=np.percentile(kap, [25, 50, 75]), ns=len(sub), sl=sl, sg=sg)
HL = {}; lines = []; lines_b = []; supp = []
for mode, lab in (('dev', 'development (seed 1)'), ('eval', 'retrospective (seeds 2, 3)'), ('conf', 'fresh confirmation')):
    R = json.load(open(os.path.join(DC, f'harmonic_{mode}.json'))); USED.add(f'diagnostics_cost/harmonic_{mode}.json')
    a = hstats(R, 'oa-vcb'); c = hstats(R, 'member'); HL[mode] = (a, c)
    lines.append(f"{lab} & {a['n']}/{a['N']} & {a['e0']:.4f} & {a['e2']:.4f} & {100 * a['R']:.1f} [{100 * a['Rci'][0]:.1f}, {100 * a['Rci'][1]:.1f}] & {a['g24']:.2f} [{a['gci'][0]:.2f}, {a['gci'][1]:.2f}] \\\\")
    lines_b.append(f"{lab} & {a['kq'][1]:.2f} [{a['kq'][0]:.2f}, {a['kq'][2]:.2f}] & {a['ns']} & {100 * a['sl']:.1f} & {100 * a['sg']:.1f} \\\\")
    for who, d in (('OA-VCB', a), ('member', c)):
        note(f"  {mode} {who}: n {d['n']}/{d['N']} e {d['e0']:.4f}->{d['e2']:.4f} R {100 * d['R']:.1f}% [{100 * d['Rci'][0]:.1f}, {100 * d['Rci'][1]:.1f}] G24 {d['g24']:.3f} "
             f"[{d['gci'][0]:.3f}, {d['gci'][1]:.3f}] kappa {d['kq']} subset {d['ns']}: local {100 * d['sl']:.1f}% global {100 * d['sg']:.1f}%")
    for sd in sorted({r['seed'] for r in R}):
        Rs = [r for r in R if r['seed'] == sd]
        for who in ('oa-vcb', 'member'):
            d = hstats(Rs, who); v = 'strong' if (d['R'] >= 0.5 and d['g24'] >= 1.25) else ('partial' if d['R'] >= 0.2 else 'little')
            supp.append(f"{lab.split(' (')[0]} & {sd} & {'OA-VCB' if who == 'oa-vcb' else 'reference'} & {d['n']} & {100 * d['R']:.1f} [{100 * d['Rci'][0]:.1f}, {100 * d['Rci'][1]:.1f}] & {d['g24']:.2f} & {v} \\\\")
write_panels('tab_harmonic', "Post-hoc harmonic enrichment of the frozen OA-VCB predictions (barriers; patch radius 2, fixed on the development set; the fresh confirmation set has 100 "
      "new operators per checkpoint). (a) Detector-positive samples / all samples; mean energy error before and after the local correction; "
      "$R = 1-\\overline{e}_{\\mathrm{after}}/\\overline{e}_{\\mathrm{before}}$ in \\% [95\\% CI]; $G_{24}$: paired ratio of the patch-indicator control's error to the harmonic "
      "correction's [95\\% CI]. (b) Per-sample captured fraction of squared energy error, median [quartiles]; on the fixed global-reference subset (number of samples), $R$ of the local "
      "and of the global correction, in \\%.", 'tab:harmonic',
      [('(a) Primary outcome', 'llllll', 'set & samples & $e$ before & $e$ after & $R$ (\\%) & $G_{24}$', lines),
       ('(b) Captured fraction and local versus global extension', 'lllll', 'set & captured fraction & subset size & local $R$ (\\%) & global $R$ (\\%)', lines_b)])
write('tab_harmonic_seeds', "Harmonic enrichment per seed, OA-VCB and classical reference: detector-positive samples, $R$ in \\% [95\\% CI], $G_{24}$, and the outcome category of the protocol.",
      'tab:harmonicseeds', 'lllllll', 'set & seed & predictor & samples & $R$ (\\%) & $G_{24}$ & outcome', supp)

# ------------------------------------------------------------------ Table 8: right-hand sides
note('== Table 8: right-hand sides')
Js = []
for s in SEEDS:
    J = json.load(open(os.path.join(DC, f'rhs_generalisation_s1_seed{s}.json'))); USED.add(f'diagnostics_cost/rhs_generalisation_s1_seed{s}.json')
    assert J['sample_ids'] == IDS; Js.append(J['loads'])
lines = []; FIXED = {}
LAB = {'f = 1 (training)': '$f=1$ (training load)', 'f = -1': '$f=-1$', 'f = 3': '$f=3$', 'session-2 random': 'random-load distribution (Sec.~\\ref{sec:k0})',
       'smooth +-': 'smooth, both signs', 'point source': 'localised source'}
for n in Js[0]:
    lr = np.concatenate([np.log(np.array(J[n]['mem']) / np.array(J[n]['net'])) for J in Js]); b = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr)))
    ci = np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5])); per = [np.exp(np.mean(np.log(np.array(J[n]['mem']) / np.array(J[n]['net'])))) for J in Js]
    uc = np.mean([np.mean(J[n]['uc']) for J in Js]); net = np.mean([np.mean(J[n]['net']) for J in Js]); mem = np.mean([np.mean(J[n]['mem']) for J in Js])
    worse = sum(int(np.sum(np.array(J[n]['net']) > np.array(J[n]['uc']) * (1 + 1e-6))) for J in Js)
    FIXED[n] = (np.exp(lr.mean()), ci, np.mean(lr > 0))
    lines.append(f"{LAB[n]} & {uc:.3f} & {fmt(net)} & {fmt(mem)} & {' / '.join(f'{p:.2f}' for p in per)} & {np.exp(lr.mean()):.2f} [{ci[0]:.2f}, {ci[1]:.2f}] & {np.mean(lr > 0):.2f} & {worse} \\\\")
    note(f"  {n}: OA-VCB {net:.3e} member {mem:.3e} G {np.exp(lr.mean()):.3f} [{ci[0]:.2f}, {ci[1]:.2f}] W {np.mean(lr > 0):.2f} per seed {[round(float(p), 2) for p in per]} worse than u_c {worse}")
write('tab_rhs', "The fixed-load ($f=1$) OA-VCB checkpoints of Table~\\ref{tab:core} applied to other loads (channels, held-out seeds 1--3, 300 operators); $u_c$ is recomputed for "
      "each load. Mean energy errors, reference/OA-VCB $G$ per seed and pooled, $W$, and the number of samples on which OA-VCB is worse than $u_c$.", 'tab:rhs', 'llllllll',
      'load & $u_c$ & OA-VCB & reference & $G$ per seed & $G$ pooled [95\\% CI] & $W$ & $>u_c$', lines, resize=True)

# ------------------------------------------------------------------ Table 8b: the random-load checkpoint on the same loads (Test B)
note('== Table 8b: random-load model on other loads')
JR = []
for s_ in SEEDS:
    J = json.load(open(os.path.join(DC, f'rhs_random_model_seed{s_}.json'))); USED.add(f'diagnostics_cost/rhs_random_model_seed{s_}.json'); assert J['sample_ids'] == IDS; JR.append(J['loads'])
lines = []; lines_b = []; lines_c = []
f2 = lambda x: f"{x:.2f}" if x < 1 else f"{x:.1f}"
for n in JR[0]:
    e, m, f, c = (np.concatenate([np.array(J[n][k]) for J in JR]) for k in ('net', 'mem', 'fixed_net', 'uc'))
    per = [G(np.array(J[n]['mem']), np.array(J[n]['net'])) for J in JR]
    g = G(m, e); gf = G(f, e)
    lab = '$f=1$' if n.startswith('f = 1') else ('$f=3$ (homogeneity check)' if n == 'f = 3' else LAB[n])
    lines.append(f"{lab} & {fmt(e.mean())} & {fmt(m.mean())} & {gtx(g)} & {g[2]:.2f} & {int(np.sum(e > c * (1 + 1e-6)))} \\\\")
    fg = FIXED[n]
    lines_c.append(f"{lab} & {fg[0]:.2f} [{fg[1][0]:.2f}, {fg[1][1]:.2f}] & {fg[2]:.2f} & {gtx(g)} & {min(p[2] for p in per):.2f} & {f2(gf[0])} \\\\")
    lines_b.append(f"{lab} & {' / '.join(f'{p[0]:.2f}' for p in per)} & {' / '.join(f'{p[2]:.2f}' for p in per)} & {fmt(f.mean())} & {f2(gf[0])} & {gf[2]:.2f} \\\\")
    note(f"  {n}: random-load {e.mean():.3e} fixed {f.mean():.3e} member {m.mean():.3e}; per-seed G {[round(float(p[0]), 2) for p in per]} W {[round(p[2], 2) for p in per]}; pooled G {gtx(g)} W {g[2]:.2f}; fixed/random G {gf[0]:.2f} W {gf[2]:.2f}")
write_panels('tab_rhs_random', "The OA-VCB checkpoints trained on the random-load distribution (Section~\\ref{sec:k0}; one model per held-out data seed 1--3) on the loads of "
      "Table~\\ref{tab:rhs}, same operators. (a) Pooled over the 300 operators: mean energy errors of the random-load models and of the reference, reference/random-model $G$ [95\\% CI] "
      "and $W$, and samples worse than $u_c$. (b) Per seed (1 / 2 / 3): reference/random-model $G$ and $W$; mean error of the fixed-load models of Table~\\ref{tab:rhs}; and pooled "
      "fixed-load/random-load $G$ and $W$ (above 1 and above 0.5 favour the random-load models). The prespecified criterion per seed and load family was $G\\ge2$ and $W\\ge0.8$.",
      'tab:rhsrandom',
      [('(a) Pooled', 'llllll', 'load & random-load & reference & $G$ [95\\% CI] & $W$ & $>u_c$', lines),
       ('(b) Per seed, and against the fixed-load models', 'llllll', 'load & $G$ per seed & $W$ per seed & fixed-load & fixed/random $G$ & $W$', lines_b)])

# 13 Oct (FINAL_SUBMISSION_REVIEW_ASTRA, section 3: "combine the main summaries"): main-text summary of the two load tables.
write('tab_loads', "Load dependence (channels, held-out data seeds 1--3, the same 300 operators for every load; $u_c$ recomputed for each load). "
      "Pooled reference/OA-VCB $G$ [95\\% CI] and $W$ for the models trained with the single load $f=1$ and for the models trained on the random-load "
      "distribution (one model per data seed; $W$: smallest of the three per-seed win fractions), and the paired ratio of the fixed-load models' error to "
      "the random-load models' (above 1 favours the random-load models). Per-seed values and mean errors: Tables~\\ref{tab:rhs} and~\\ref{tab:rhsrandom}.",
      'tab:loads', 'llllll', 'load & fixed-load $G$ [95\\% CI] & $W$ & random-load $G$ [95\\% CI] & $W$ (min) & fixed/random', lines_c)

# ------------------------------------------------------------------ Table 9: cost, parsed from the timing logs
note('== Table 9: cost')
def parse(log):
    T = open(log).read(); USED.add(os.path.relpath(log, ROOT)); d = {}
    pats = dict(uc=r'u_c = level-\d Galerkin factor \+ solve:\s+([\d.]+)', od=r'operator data \(hierarchy.*?\)\s+([\d.]+)', pat=r'\+ (\d+) ms once per mesh',
                i8=r'inference batch 8 \(torch CPU\)\s+([\d.]+)', i1=r'inference batch 1 \(torch CPU\)\s+([\d.]+)', tot=r'TOTAL = .*?\)\s+[\d.e+-]+\s+([\d.]+)',
                mem=r'classical member, one pass .*?\s+[\d.e+-]+\s+([\d.]+)', tg=r'two-grid PCG .*? to oa-vcb error\s+\(matched\)\s+([\d.]+)',
                tgi=r'two-grid PCG .*?its median (\d+) \(range (\d+-\d+)\)', amg=r'SA-AMG PCG .*? to oa-vcb error\s+\(matched\)\s+([\d.]+)',
                amgi=r'SA-AMG PCG .*?its median (\d+) \(range (\d+-\d+)\)', vc=r'V\(4,4\) preconditioner, to oa-vcb error\s+\(matched\)\s+([\d.]+)',
                vci=r'V\(4,4\) preconditioner.*?its median (\d+) \(range (\d+-\d+)\)', dir=r'direct \(SuperLU factor \+ solve, exact\)\s+\S+\s+([\d.]+)', asm=r'fine-matrix assembly, common to all methods \(excluded\)\s+([\d.]+)')
    for k, p in pats.items():
        m = re.search(p, T)
        if m is None: raise ValueError(f"{log}: no match for {k}")
        d[k] = m.groups() if len(m.groups()) > 1 else m.group(1)
    return d
C3 = parse(os.path.join(DC, 'timing_main_L3_k02_seed1_v11oct.log')); C4 = parse(os.path.join(DC, 'timing_main_L4_k03_seed1_v11oct.log'))
f1 = lambda x: f"{float(x):,.1f}".replace(',', '{,}')
def crow(lab, k, extra=None):
    cells = []
    for C in (C3, C4):
        v = f1(C[k])
        if extra == 'i1': v += f" ({f1(C['i1'])})"
        elif extra: v += f" [{C[extra][0]}; {C[extra][1].replace('-', '--')}]"
        cells.append(v)
    return f"{lab} & " + " & ".join(cells) + " \\\\"
lines = ["\\multicolumn{3}{l}{\\emph{OA-VCB (one pass)}} \\\\", crow('$u_c$ (level-$k_0$ Galerkin factor and solve)', 'uc'),
         crow('operator data (hierarchy, Galerkin levels, $Z$)', 'od'), crow('inference, batch 8 (batch 1)', 'i8', 'i1'), crow('\\textbf{total} (batch 8)', 'tot'),
         "\\midrule\\multicolumn{3}{l}{\\emph{single classical pass, error not matched (10 times larger)}} \\\\", crow('classical reference, from $u_c$', 'mem'),
         "\\midrule\\multicolumn{3}{l}{\\emph{iterative, run to OA-VCB's error on each sample [median; range of iterations]}} \\\\",
         crow('two-grid PCG', 'tg', 'tgi'), crow('SA-AMG PCG', 'amg', 'amgi'), crow('PCG with the classical V(4,4)', 'vc', 'vci'),
         "\\midrule\\multicolumn{3}{l}{\\emph{direct, to solver accuracy}} \\\\", crow('sparse direct solve (SuperLU)', 'dir')]
note(f"  L3: total {C3['tot']} ms vs TG-PCG {C3['tg']} / AMG {C3['amg']} / direct {C3['dir']} / member {C3['mem']} / V(4,4)-PCG {C3['vc']};  L4: {C4['tot']} vs {C4['tg']} / {C4['amg']} / {C4['dir']} / {C4['mem']} / {C4['vc']}")
write('tab_cost', "Cost per test sample on one CPU (2 threads), channels, held-out seed 1, in ms. Every row starts from the assembled fine matrix; its assembly (" + f1(C3['asm']) + " / " + f1(C4['asm']) + " ms, "
      "common to all methods) is not included. Per-operator set-up is included in every row; the sparsity patterns of the hierarchy, "
      "computed once per mesh (" + C3['pat'] + " / " + C4['pat'] + " ms), and training are not. Classical rows: mean over the 100 samples of the per-sample minimum over 3 (L3) / 2 (L4) "
      "repetitions; OA-VCB operator data and inference: minimum over the repetitions of the whole-test-set time divided by 100. Iteration counts were determined beforehand, untimed, "
      "with the reference solution (a matched-accuracy benchmark, not a stopping criterion).", 'tab:cost', 'lll', ' & L3, $k_0=2$ (3{,}486 unknowns) & L4, $k_0=3$ (14{,}398)', lines)

# ------------------------------------------------------------------ elasticity (Session 4; member recomputed on CPU)
note('== Table: plane-strain elasticity (one data set, preliminary)')
EJ = json.load(open(os.path.join(ROOT, 'diagnostics_elasticity', 'elast_member.json'))); USED.add('diagnostics_elasticity/elast_member.json')
if EJ['sample_ids'] != IDS: raise ValueError('elasticity: sample ids')
erec = [r for r in REC if r['c'].get('field') == 'bands' and r['c'].get('contrast') == 100.0 and r['c'].get('level') == 3]
def eget(model, seed):
    h = [r for r in erec if r['model'] == model and r['c'].get('seed') == seed]
    if len({tuple(np.round(x['en'], 12)) for x in h}) != 1: raise ValueError(f"elasticity {model} seed {seed}: {len(h)} records")
    check_ids(h[0], f"elasticity {model}"); USED.add(h[0]['file']); return h[0]
ev = eget('oa-vcb', 0); evc = ev['en']; emem = np.array(EJ['member']); epl = np.array(EJ['plain'])
if np.max(np.abs(np.array(EJ['oa_vcb']) / evc - 1)) > 1e-12: raise ValueError('elasticity: OA-VCB record mismatch')
if np.max(np.abs(epl / eget('classical V(4,4) op.-adapted', 0)['en'] - 1)) > 1e-5: raise ValueError('elasticity: plain cycle not reproduced')
lines = []
def erow(lab, e, params='--', ref=True, is_mem=False):
    g = G(emem, e)
    lines.append(f"{lab} & {params} & {fmt(np.mean(e))} & " + ("1 & --" if is_mem else f"{gtx(g)} & {g[2]:.2f}") + " \\\\")
erow('OA-VCB (ours)', evc, f"{ev['res']['params']:,}".replace(',', '{,}'))
for key, lab in (('oa-ll-op', 'OA-Lift'), ('gnn-op', 'GNN-op')):
    rs = [eget(key, sd) for sd in (0, 1, 2)]
    for sd, r in zip((0, 1, 2), rs): erow(f"{lab}, init.\\ seed {sd}", r['en'], f"{r['res']['params']:,}".replace(',', '{,}'))
lines.append('\\midrule')
erow('classical reference', emem, is_mem=True); erow('classical V(4,4), no proj.', epl); erow('$u_c$ alone', np.array(EJ['uc']))
its = np.array(EJ['pcg_its']); g = G(emem, evc)
note(f"  OA-VCB {evc.mean():.4e} member {emem.mean():.4e} plain {epl.mean():.4e} uc {np.mean(EJ['uc']):.4e}; member/OA-VCB G {gtx(g)} W {g[2]:.2f}; "
     f"TG-PCG its median {np.median(its):.0f} range {its.min()}-{its.max()}")
write('tab_elasticity', "Plane-strain elasticity (stiff bands, $E_f/E_m=100$, traction-free pores; one data set, test samples 300--399; preliminary). Mean energy error, "
      "and paired reference/model $G$ [95\\% CI] and $W$. OA-VCB has one training run; the two baselines were trained with three initialisation seeds on the same data. "
      f"Two-grid PCG first reaches OA-VCB's per-sample energy error after a median of {np.median(its):.0f} iterations (range {its.min()}--{its.max()}).",
      'tab:elasticity', 'lrlll', 'model & params & energy & $G$ [95\\% CI] & $W$', lines)

# ------------------------------------------------------------------ contained classical cycles (Jacobi; fixed-interval Chebyshev) vs the adaptive reference
note('== Table: contained classical cycles and the adaptive Chebyshev reference')
la, lb = [], []
def crow2(lab, e_vcb, ec, ef, ej):
    gf, gj, gfc, gjc = G(ef, e_vcb), G(ej, e_vcb), G(ef, ec), G(ej, ec)
    la.append(f"{lab} & {fmt(e_vcb.mean())} & {fmt(ec.mean())} & {fmt(ef.mean())} & {fmt(ej.mean())} & {gfc[0]:.3f} & {gjc[0]:.2f} \\\\")
    lb.append(f"{lab} & {G(ec, e_vcb)[0]:.2f} & {gtx(gf)} & {gf[2]:.2f} & {gtx(gj)} & {gj[2]:.2f} \\\\")
    note(f"  {lab}: ref/OA-VCB {G(ec, e_vcb)[0]:.2f}; fixedCheb/OA-VCB {gtx(gf)} W {gf[2]:.2f}; Jacobi/OA-VCB {gtx(gj)} W {gj[2]:.2f}; fixedCheb/ref {gfc[0]:.3f} (ref better on {gfc[2]:.2f}); Jacobi/ref {gjc[0]:.2f} (ref better on {gjc[2]:.2f})")
def jload(tag, cheb_rec):
    J = json.load(open(os.path.join(DC, f'jacobi_member_{tag}.json'))); USED.add(f'diagnostics_cost/jacobi_member_{tag}.json')
    F = json.load(open(os.path.join(DC, f'fixed_cheb_{tag}.json'))); USED.add(f'diagnostics_cost/fixed_cheb_{tag}.json')
    if J['sample_ids'] != IDS or F['sample_ids'] != IDS: raise ValueError(f'{tag}: ids')
    ec = np.array(J['chebyshev'])
    for x, nm in ((ec, 'jacobi_member chebyshev'), (np.array(F['adaptive']), 'fixed_cheb adaptive')):
        d = np.max(np.abs(x / cheb_rec - 1))
        if d > 2e-5: raise ValueError(f'{tag}: {nm} does not reproduce the stored reference ({d:.1e})')
    if np.max(np.abs(np.array(F['jacobi']) / np.array(J['jacobi']) - 1)) > 1e-10: raise ValueError(f'{tag}: Jacobi arrays differ')
    return ec, np.array(F['fixed3']), np.array(J['jacobi'])
for sd in SEEDS:
    crow2(f"channels, seed {sd}", get('oa-vcb', **CH, data_seed=sd)['en'], *jload(f"ch_s{sd}", classical(CH, sd)[0][MEM]))
for sd in SEEDS:
    crow2(f"barriers, seed {sd}", get('oa-vcb', **BA, data_seed=sd)['en'], *jload(f"ba_s{sd}", classical(BA, sd)[0][MEM]))
for sd in SEEDS:
    k0c_ = {json.loads(l)['model']: json.loads(l) for l in open(os.path.join(DC, f'k0_0_classical_seed{sd}.jsonl'))}
    crow2(f"$k_0=0$, seed {sd}", np.array(rv[sd]['result']['en_s']), *jload(f"k00_s{sd}", np.array(k0c_['V(4,4) op.-adapted + defl. + proj., from 0']['en_s'])))
crow2("L4, $k_0=3$, seed 1", get('oa-vcb', level=4, coarse='perforated', field='channels', contrast=1e4, k0=3, data_seed=1)['en'], *jload("l4_s1", c4[MEM]))
EJ2 = json.load(open(os.path.join(ROOT, 'diagnostics_elasticity', 'elast_member.json')))
crow2("elasticity (prelim.)", evc, np.array(EJ2['member']), np.array(EJ2['fixed3']), np.array(EJ2['jacobi']))
write_panels('tab_contained', "Classical cycles contained in the architecture (Proposition~\\ref{prop:containment}, same hierarchy, coarse solution, deflation and Galerkin output; "
      "$\\nu=4$): fixed-interval Chebyshev smoothing ($\\lambda=3$, same weights on every level) and weighted Jacobi ($\\omega=2/3$), together with the adaptive Chebyshev reference used in the other "
      "tables. The contained cycles are evaluated in double precision with the unregularised Galerkin solves of the mathematical construction; the regularisation and finite-precision "
      "safeguards of the deployed network are described in Section~\\ref{sec:properties}. (a) Mean energy errors and paired $G$ of each contained cycle over the reference. (b) Paired $G$ of "
      "each classical cycle over OA-VCB ([95\\% CI] for the contained cycles) and OA-VCB's win fraction $W$. Elasticity: no deflation; one data set.", 'tab:contained',
      [('(a) Mean energy errors', 'lllllll', 'set & OA-VCB & reference & fixed Cheb. & Jacobi & fixed Cheb./ref.\\ $G$ & Jacobi/ref.\\ $G$', la),
       ('(b) Gains of OA-VCB', 'llllll', 'set & ref./OA-VCB $G$ & fixed Cheb./OA-VCB $G$ & $W$ & Jacobi/OA-VCB $G$ & $W$', lb)])

# ------------------------------------------------------------------ appendix: excluded runs, bridge oracle
lines = []
for key, lab, arch in (('wno-off', 'WNO, published architecture', 'Tripura--Chakraborty v2.0.0 port, pytorch\\_wavelets'), ('mgno-off', 'MgNO, published size', 'official model code, width 24')):
    r = get(key, **CH, data_seed=1, params=80000); res = r['res']
    lines.append(f"{lab} & {arch} & {res.get('params'):,}".replace(',', '{,}') + f" & {fmt(r['en'].mean())} & {fmt(np.nanmean(r['rl2']))} & {res.get('n_worse_than_uc')}/100 \\\\")
cu = classical(CH, 1)[0]['coarse solve alone']; cur = classical(CH, 1)[1]['coarse solve alone']
lines.append(f"$u_c$ alone (reference) & -- & -- & {fmt(cu.mean())} & {fmt(cur.mean())} & -- \\\\")
write('tab_excluded', "Published-size grid models trained under the protocol (channels, seed 1, same loss and schedule, no parameter cap) and excluded from superiority claims: "
      "mean energy error, nodal relative $\\ell^2$ error, and samples worse than $u_c$ in energy.", 'tab:excluded', 'llllll',
      'model & implementation & params & energy & nodal rel.\\ $\\ell^2$ & worse than $u_c$', lines, resize=True)
T = open(os.path.join(DC, 'grid_floor_exact.log')).read(); USED.add('diagnostics_cost/grid_floor_exact.log'); lines = []
for blk in re.split(r'\n(?=ng )', T.strip()):
    m = re.match(r'ng (\d+): rank\(T\) (\d+) of (\d+)', blk)
    if not m: continue
    rt = re.search(r'round trip\s+energy/\|\|t\|\| ([\d.e+-]+).*?full-solution metric ([\d.e+-]+)', blk)
    br = re.search(r'best range\s+energy/\|\|t\|\| ([\d.e+-]+).*?full-solution metric ([\d.e+-]+)', blk)
    lines.append(f"${m.group(1)}\\times{m.group(1)}$ & {m.group(2)} of {m.group(3)} & {float(rt.group(1)):.2f} & {float(rt.group(2)):.2f} & {float(br.group(1)):.2e} & {float(br.group(2)):.2e} \\\\")
write('tab_bridge', "The grid bridge on 40 training operators (channels, seed 1): numerical rank of the grid-to-mesh map $T$ (eigenvalues of $TT^\\top$ above $10^{-12}$ times the largest); energy error of the round trip of the exact correction "
      "$t=u-u_c$ (mesh $\\to$ grid by piecewise-linear interpolation, grid $\\to$ mesh by bilinear interpolation), relative to $\\|t\\|_A$ and to $\\|u\\|_A$; and of the best "
      "approximation of $t$ in the numerically resolved range of $T$ (Galerkin solve on an orthonormal basis of that range). Mean relative energy errors over the samples.", 'tab:bridge', 'llllll',
      'grid & rank($T$) & round trip / $\\|t\\|_A$ & round trip / $\\|u\\|_A$ & best / $\\|t\\|_A$ & best / $\\|u\\|_A$', lines)

# 11 Oct (independent reviews): sensitivity of the detector thresholds and of the number of transfer Jacobi steps, with the
# classical reference (diagnostics_cost/sensitivity_detector_transfers.py) and the learned barrier checkpoints re-evaluated with
# theta = 0.05 (diagnostics_cost/detector_theta_learned.py).
SN = json.load(open(os.path.join(DC, 'sensitivity_detector_transfers.json'))); USED.add('diagnostics_cost/sensitivity_detector_transfers.json')
TL = json.load(open(os.path.join(DC, 'detector_theta_learned.json'))); USED.add('diagnostics_cost/detector_theta_learned.json')
def frac(rows, k): a = [list(map(int, r[k].split('/'))) for r in rows]; return f"{sum(x for x, _ in a)}/{sum(y for _, y in a)}"
linesA = []
for th in (0.05, 0.1, 0.2):
    for tau in (0.02, 0.05, 0.1):
        rows = [SN[f"A_ds{d}_th{th}_tau{tau}"] for d in (1, 2, 3)]
        eall = np.mean([np.mean(r['en_s']) for r in rows]); eis = np.sum([r['ref_islands'] * r['n_island_samples'] for r in rows]) / np.sum([r['n_island_samples'] for r in rows])
        mark = ' (used)' if (th, tau) == (0.1, 0.05) else ''
        linesA.append(f"{th:g} & {tau:g}{mark} & {frac(rows, 'recall')} & {frac(rows, 'precision')} & {fmt(eall)} & {fmt(eis)} \\\\")
linesL = []
for th in (0.1, 0.05):
    r = [TL[f"ds{d}_theta{th}"] for d in (1, 2, 3)]
    ev = np.concatenate([x['en_vcb'] for x in r]); er = np.concatenate([x['en_ref'] for x in r])
    g = G(er, ev)
    linesL.append(f"{th:g} & {fmt(ev.mean())} & {fmt(er.mean())} & {gtx(g)} & {np.mean(ev < er):.2f} \\\\")
linesB = []
for m in (0, 1, 2, 4, 8):
    c, b = SN[f"B_channels_m{m}"], SN[f"B_barriers_m{m}"]
    linesB.append(f"{m}{' (used)' if m == 4 else ''} & {c['op_complexity']:.2f} & {c['build_ms']:.0f} & {fmt(c['ref_mean'])} & {c['G_vs_m4']:.2f} & {fmt(b['ref_mean'])} & {b['G_vs_m4']:.2f} \\\\")
write_panels('tab_sensitivity', "Sensitivity of two fixed choices (classical reference unless stated; L3, $k_0=2$). (a) Detector thresholds on the barrier "
             "test sets of held-out seeds 1--3 (300 operators, 202 coefficient islands in total; an operator may contain several). Recall: islands whose "
             "indicator lies in $\\operatorname{span}Z$ (relative Euclidean distance below $10^{-8}$), out of all islands. Precision: detected vectors "
             "with at least $90\\%$ of their support inside one island, out of all detected vectors (no one-to-one matching, so the two numerators "
             "count different objects and need not agree). Mean reference energy error over all samples and over the island samples. (b) The trained "
             "barrier OA-VCB models (trained with $\\theta=0.1$) evaluated with $Z$ from $\\theta=0.1$ and $0.05$ ($\\tau=0.05$), pooled seeds 1--3: "
             "mean energy errors, reference/OA-VCB $G$ [95\\% CI] and $W$. (c) Number $m$ of Jacobi steps in the transfers, held-out seed 1: "
             "operator complexity $\\sum_k\\operatorname{nnz}(A_k)/\\operatorname{nnz}(A_L)$ (median), hierarchy build time (median, one CPU), "
             "reference energy error and the paired geometric-mean ratio of its errors to those at $m=4$; the learned cycle was trained only "
             "with $m=4$ and $m=0$ (Table~\\ref{tab:transfers}).", 'tab:sensitivity',
             [("(a) detector thresholds, classical reference", 'llllll', '$\\theta$ & $\\tau$ & recall & precision & energy, all & energy, island samples', linesA),
              ("(b) trained OA-VCB with the detector of a smaller $\\theta$", 'lllll', '$\\theta$ & OA-VCB & reference & $G$ [95\\% CI] & $W$', linesL),
              ("(c) Jacobi steps in the transfers, classical reference", 'lllllll', '$m$ & op.\\ complexity & build (ms) & channels & $G$ vs $m=4$ & barriers & $G$ vs $m=4$', linesB)])

# ------------------------------------------------------------------ appendix: published Darcy benchmark (Session 9)
# 12 Oct.  kaggle_runs/published_darcy.py on the FNO piecewise-constant Darcy files (85 x 85, 1000 train / 100 test, 500 epochs),
# records Results_kaggle_session 9/published_darcy_<model>.json.  Published values are literature constants (not results of
# this study): FNO 1.08 % (Li et al. 2021; also WNO paper Table 3), WNO 0.84 % (Tripura & Chakraborty, Table 3: db4, level 4,
# width 64, 1000 / 100, 500 epochs), Transolver 0.57 % (Wu et al. 2024, Table 2: 1000 / 200 samples, loss with a gradient
# term).  The MgNO paper reports a different "Darcy smooth" data set, so no value is listed for it.
note('== Table: published Darcy benchmark (Session 9)')
PD = os.path.join(ROOT, 'Results_kaggle_session 9')
PUB = {'fno': 1.08, 'fno-official': 1.08, 'wno-db4': 0.84, 'wno-db6': 0.84, 'transolver': 0.57}
PDROWS = [('fno-official', 'FNO-port', '--'), ('fno', 'FNO-var', 'FNO-op'), ('mgno', 'MgNO-off', 'MgNO, MgNO-op'),
          ('transolver', 'Transolver-re', 'Transolver-op'), ('wno-db4', 'WNO-db4', '--'), ('wno-db6', 'WNO-db6', 'excl.'),
          ('wno-light', 'WNO-light', 'WNO-op')]
if os.path.isdir(PD):
    lines = []
    for key, lab, rows in PDROWS:
        f = os.path.join(PD, f'published_darcy_{key}.json'); J = json.load(open(f)); USED.add(f'Results_kaggle_session 9/published_darcy_{key}.json')
        e = np.array(J['test_rel_l2_s']) * 100
        if len(e) != 100 or abs(e.mean() / (100 * J['test_rel_l2']) - 1) > 1e-6: raise ValueError(f'{key}: per-sample errors do not reproduce the stored mean')
        pub = PUB.get(key); ratio = f"{e.mean() / pub:.2f}" if pub else '--'
        note(f"  {key}: {J['params']} params, test rel L2 {e.mean():.3f} % (median {np.median(e):.3f} %), published {pub}, ratio {ratio}, train {J['train_s']:.0f} s")
        lines.append(f"{lab} & {rows} & " + f"{J['params']:,}".replace(',', '{,}') + f" & {e.mean():.2f} ({np.median(e):.2f}) & " + (f"{pub:.2f}" if pub else '--') + f" & {ratio} \\\\")
    write('tab_published', "Baselines on the published piecewise-constant Darcy benchmark of \\cite{li2021fno} ($85\\times85$, 1000 training and 100 test "
          "samples, 500 epochs, one training run each): mean (median) relative $L^2$ test error in per cent, the value published for the same "
          "architecture, and their ratio. Implementations: FNO-port, faithful port of \\cite{li2021fno}; FNO-var, the simplified variant behind our FNO-op "
          "rows; MgNO-off, the official MgNO model classes \\cite{he2024mgno}; Transolver-re, our re-implementation of \\cite{wu2024transolver}; "
          "WNO-db4 and WNO-db6, ports of version 2.0.0 of the authors' WNO code with the db4 transform of the paper and the db6 transform of the "
          "repository; WNO-light, our light WNO layer. Published values: FNO \\cite{li2021fno}; WNO, db4 with four levels \\cite{tripura2023wno}; Transolver "
          "\\cite{wu2024transolver}, obtained with 200 test samples and an additional gradient term in the loss. The MgNO paper \\cite{he2024mgno} "
          "reports a different smooth Darcy data set, so no value is listed. ``Used in'': the rows of Tables~\\ref{tab:channels} and~\\ref{tab:barriers} "
          "(or the excluded run of Table~\\ref{tab:excluded}) that use the same implementation.", 'tab:published', 'llrlll',
          'implementation & used in & params & rel.\\ $L^2$ (\\%) & published (\\%) & ratio', lines, resize=True)
else:
    note('  Results_kaggle_session 9 not present: tab_published not written')

# ------------------------------------------------------------------ refinement without retraining (Session 11)
# 13 Oct.  L5: zs_L5.json (per-sample record, 100 test samples); paired with the L4 record of the same coefficient draws.
# L6 (batch size 1): the run was stopped by the platform's session limit; its per-sample log lines are the record
# (zs_L6.log, and the earlier partial copy *.txt whose lines carry a "zs_L6" prefix); four significant digits.
note('== Refinement without retraining (Session 11)')
D11 = os.path.join(ROOT, 'Results_kaggle_session 11')
if os.path.isdir(D11):
    f5 = os.path.join(D11, 'zs_L5.json')
    if os.path.exists(f5):
        USED.add('Results_kaggle_session 11/zs_L5.json'); J5 = json.load(open(f5))['zs_L4toL5_k4']
        if J5['ids'] != IDS: raise ValueError('zs_L5.json: ids')
        v5, r5, u5 = (np.array(J5[k]) for k in ('en_vcb', 'en_ref', 'en_uc')); g5 = G(r5, v5)
        spec4 = dict(level=4, coarse='perforated', field='channels', contrast=1e4, k0=3)
        v4 = get('oa-vcb', **spec4, data_seed=1)['en']; r4 = classical(spec4, 1)[0][MEM]
        dlog = np.log(r5 / v5) - np.log(r4 / v4); bb = np.random.default_rng(0).integers(0, len(dlog), (5000, len(dlog)))
        note(f"  L5 ({J5['unknowns']} unknowns, k0 {J5['k0']}): OA-VCB {v5.mean():.4e} reference {r5.mean():.4e} u_c {u5.mean():.4e}; G {gtx(g5)} W {g5[2]:.2f}; "
             f"worse than u_c {int(np.sum(v5 > u5 * (1 + 1e-9)))}; smallest ratio {np.min(r5 / v5):.2f}; logged s/sample {J5['s_per_sample']:.1f}")
        note(f"  paired gain ratio L5/L4 (same coefficient draws): {np.exp(dlog.mean()):.3f} [{np.exp(np.percentile(dlog[bb].mean(1), 2.5)):.3f}, "
             f"{np.exp(np.percentile(dlog[bb].mean(1), 97.5)):.3f}]; L4 G {G(r4, v4)[0]:.2f}; gain smaller at L5 on {np.mean(dlog < 0):.2f} of samples")
    L6 = {}
    for f in sorted(glob.glob(os.path.join(D11, '*.txt')) + glob.glob(os.path.join(D11, 'zs_L6*.log'))):
        USED.add(os.path.relpath(f, ROOT)); pref = f.endswith('.txt')
        for l in open(f):
            m = re.search((r'zs_L6 .*' if pref else r'^\s*') + r'samples (\d+)-(\d+): OA-VCB ([\d.e+-]+)\s+reference ([\d.e+-]+)\s+\(([\d.]+) s/sample', l)
            if not m: continue
            i0, i1 = int(m.group(1)), int(m.group(2))
            if i0 != i1: raise ValueError('L6 log lines must be single samples (ZS_BS=1)')
            rec = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
            if i0 in L6 and L6[i0] != rec: raise ValueError(f'L6 sample {i0}: conflicting log lines')
            L6[i0] = rec
    if L6:
        ids6 = sorted(L6)
        if ids6 != list(range(300, 300 + len(ids6))): raise ValueError(f'L6 samples not contiguous from 300: {ids6}')
        v6 = np.array([L6[i][0] for i in ids6]); r6 = np.array([L6[i][1] for i in ids6]); t6 = np.array([L6[i][2] for i in ids6]); g6 = G(r6, v6)
        note(f"  L6 (per-sample log lines): {len(ids6)} samples (ids {ids6[0]}-{ids6[-1]}), OA-VCB {v6.mean():.4e} reference {r6.mean():.4e}, G {gtx(g6)} W {g6[2]:.2f}, "
             f"smallest ratio {np.min(r6 / v6):.2f}, logged s/sample median {np.median(t6):.0f} (operator data + inference only)")
        if os.path.exists(f5):
            sel = [IDS.index(i) for i in ids6]; d65 = np.log(r6 / v6) - np.log(r5[sel] / v5[sel]); bb = np.random.default_rng(0).integers(0, len(d65), (5000, len(d65)))
            note(f"  paired gain ratio L6/L5 on the same {len(ids6)} draws: {np.exp(d65.mean()):.3f} [{np.exp(np.percentile(d65[bb].mean(1), 2.5)):.3f}, "
                 f"{np.exp(np.percentile(d65[bb].mean(1), 97.5)):.3f}]; L5 G on these draws {G(r5[sel], v5[sel])[0]:.2f}")
else:
    note('  Results_kaggle_session 11 not present')

# ------------------------------------------------------------------ stronger classical smoothers, one CPU (12 Oct)
# diagnostics_cost/accuracy_vs_time.py (channels, barriers; held-out seed 1; one machine, OA-VCB re-timed on it).
note('== Table: stronger classical smoothers and repeated classical passes (accuracy_vs_time)')
AV = {}
for fam, fn in (('channels', 'accuracy_vs_time.json'), ('barriers', 'accuracy_vs_time_barriers.json')):
    f = os.path.join(DC, fn)
    if os.path.exists(f): AV[fam] = json.load(open(f)); USED.add(f'diagnostics_cost/{fn}')
if len(AV) == 2:
    ROWSAV = [('reference, Chebyshev nu=4 (paper)', 'reference, Chebyshev $\\nu=4$ (Table~\\ref{tab:core})'),
              ('reference, Chebyshev nu=8', 'reference, Chebyshev $\\nu=8$'), ('reference, Chebyshev nu=16', 'reference, Chebyshev $\\nu=16$'),
              ('reference, symmetric Gauss-Seidel 4 sweeps', 'reference, symmetric Gauss--Seidel, 4 sweeps'),
              ('reference nu=4, applied 2 times', 'reference ($\\nu=4$) applied twice'), ('reference nu=4, applied 3 times', 'reference ($\\nu=4$) applied 3 times')]
    lines = []
    for fam in ('channels', 'barriers'):
        M = AV[fam]['methods']; v = np.array(M['OA-VCB (one pass)']['energy_s'])
        e0 = get('oa-vcb', **(CH if fam == 'channels' else BA), data_seed=1)['en']
        # CPU re-evaluation vs the T4 record: <= 5.6e-6 on channels; on barriers one sample at 1.41e-3 (median 3e-6), the documented
        # CPU/GPU round-off envelope of the deflated, projected output (Appendix C, amendment v3).
        if np.max(np.abs(v / e0 - 1)) > 2e-3: raise ValueError(f'accuracy_vs_time {fam}: OA-VCB errors do not reproduce the record')
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{fam}}}: OA-VCB {fmt(v.mean())}, {M['OA-VCB (one pass)']['ms_mean']:.0f}\\,ms}} \\\\")
        for key, lab in ROWSAV:
            e = np.array(M[key]['energy_s']); g = G(e, v)
            note(f"  {fam} {key}: {e.mean():.3e}, {M[key]['ms_mean']:.1f} ms, G(cls/OA-VCB) {gtx(g)} W {g[2]:.2f}")
            lines.append(f"\\quad {lab} & {fmt(e.mean())} & {gtx(g)} & {g[2]:.2f} & {M[key]['ms_mean']:.0f} \\\\")
    write('tab_avt', "Stronger classical smoothers and repeated classical passes, held-out seed 1, measured on one CPU "
          f"({AV['channels']['cpu']}, {AV['channels']['threads']} threads; a different machine from Table~\\ref{{tab:cost}}, so OA-VCB was re-timed): "
          "energy, the arithmetic mean relative energy error over the 100 test samples (the quantity plotted in Figure~\\ref{fig:avt}); $G$, the paired "
          "geometric-mean ratio of each classical variant's error to OA-VCB's [95\\% CI] (below 1: the classical variant is more accurate in this metric); "
          "$W$, the fraction of samples on which OA-VCB is more accurate; and the mean time per sample from $u_c$ with set-up, in ms (classical rows: "
          "per-sample minimum over the repetitions; OA-VCB: operator data and batched inference, minimum over the repetitions of the whole-test-set time "
          "divided by the number of samples). The eight- and sixteen-step rows exceed the residual-polynomial degree covered by "
          "Proposition~\\ref{prop:containment} for the four-stage architecture; no learned model with that larger budget was trained.", 'tab:avt', 'lllll',
          'method & energy & $G$ [95\\% CI] & $W$ & ms', lines)
else:
    note('  accuracy_vs_time records not present')

# ------------------------------------------------------------------ initialisation seeds and m = 8 (Session 10)
# 12 Oct (REVIEW_AND_PURPOSE_ASTRA, item 3).  Aggregated separately from the seed-0 tables above.  Rule against selective
# reporting: if ANY Session-10 record is present, EVERY expected record must be present (otherwise this raises); if none is
# present the section is skipped and says so.  Spread across training runs is reported separately from test-operator
# uncertainty: three trained models evaluated on the same 100 operators are not 300 independent test observations.
note('== Initialisation seeds (Session 10) and m = 8 transfers')
INIT = (1, 2)
def has(model, spec, ds, **kw):
    try: get(model, **spec, data_seed=ds, **kw); return True
    except KeyError: return False
expected = ([('oa-vcb', CH, d, dict(seed=s)) for d in SEEDS for s in INIT] + [('oa-vcb', BA, d, dict(seed=s)) for d in SEEDS for s in INIT]
            + [('oa-vcbl', CH, 1, dict(seed=s)) for s in INIT] + [('oa-vcb', CH, 1, dict(oa_steps=8)), ('oa-vcb', BA, 1, dict(oa_steps=8))])
present = [has(m, sp, d, **kw) for m, sp, d, kw in expected]
if not any(present):
    note('  Session 10 records not present: tab_initseeds / tab_m8 not written (pending)')
else:
    if not all(present):
        raise ValueError('Session 10 incomplete: missing ' + str([(m, d, kw) for (m, sp, d, kw), ok in zip(expected, present) if not ok]))
    lines = []
    for nm, spec in (('channels', CH), ('barriers', BA)):
        for d in SEEDS:
            ref = classical(spec, d)[0][MEM]; gs, ms = [], []
            for s in (0,) + INIT:
                e = get('oa-vcb', **spec, data_seed=d, seed=s)['en']; g = G(ref, e); gs.append(g); ms.append(e.mean())
            note(f"  {nm} data seed {d}: G per init seed " + ", ".join(f"{s}: {gtx(g)} W {g[2]:.2f}" for s, g in zip((0,) + INIT, gs))
                 + f"; mean energy {[fmt(m) for m in ms]}")
            lines.append(f"{nm} & {d} & " + " & ".join(f"{g[0]:.2f} ({g[2]:.2f})" for g in gs) + f" & {fmt(min(ms))}--{fmt(max(ms))} \\\\")
        if nm == 'channels': lines.append('\\midrule')
    vcb = {s: get('oa-vcb', **CH, data_seed=1, seed=s)['en'] for s in (0,) + INIT}
    rbl = {0: get('oa-vcbl', **CH, data_seed=1, prefer='linear_controls')}
    rbl.update({s: get('oa-vcbl', **CH, data_seed=1, seed=s) for s in INIT})
    refc = classical(CH, 1)[0][MEM]; lines_b = []
    for s in (0,) + INIT:
        res = rbl[s]['res']; v = rbl[s]['en']
        plain = res['energy'] if 'energy_proj' in res else res.get('energy_plain'); proj = res.get('energy_proj', res['energy'])
        g1, g2 = G(refc, v), G(v, vcb[s])
        note(f"  channels seed 1, init seed {s}: reference/OA-VCBL {gtx(g1)} W {g1[2]:.2f}; OA-VCBL/OA-VCB {gtx(g2)} W {g2[2]:.2f}; "
             f"OA-VCBL mean energy projected {proj:.6g} plain {plain:.6g} ({rbl[s]['file']})")
        lines_b.append(f"{s} & {fmt(proj)} & {fmt(plain)} & {g1[0]:.2f} ({g1[2]:.2f}) & {g2[0]:.2f} ({g2[2]:.2f}) \\\\")
    write_panels('tab_initseeds', "Repeated training with initialisation seeds 0 (the runs of the other tables), 1 and 2 on the same data. (a) OA-VCB: "
          "reference/OA-VCB $G$ ($W$) per data seed and initialisation seed, and the range of the mean energy error over the three trained models. "
          "(b) The residual-linear control OA-VCBL on channels, data seed 1: mean energy error of its projected (deployed) and plain cycle output, "
          "reference/OA-VCBL $G$ ($W$), and OA-VCBL/OA-VCB $G$ ($W$) with OA-VCB of the same initialisation seed (the two factors of "
          "Section~\\ref{sec:ablation}). The ranges describe variability across training runs; the three models are evaluated on the same 100 test "
          "operators, so they are not independent test observations.", 'tab:initseeds',
          [('(a) OA-VCB', 'llllll', 'class & data seed & init.\\ 0 & init.\\ 1 & init.\\ 2 & mean energy (range)', lines),
           ('(b) OA-VCBL, channels, data seed 1', 'lllll', 'init.\\ seed & projected & plain & ref./OA-VCBL & OA-VCBL/OA-VCB', lines_b)])
    lines = []
    for nm, spec in (('channels', CH), ('barriers', BA)):
        e8 = get('oa-vcb', **spec, data_seed=1, oa_steps=8)['en']; e4 = get('oa-vcb', **spec, data_seed=1)['en']
        r8 = classical(spec, 1, oa_steps=8)[0][MEM]; r4 = classical(spec, 1)[0][MEM]
        g88, g48, g44, gr = G(r8, e8), G(r4, e8), G(r4, e4), G(r4, r8)
        note(f"  m=8 {nm}: ref(m8)/OA-VCB(m8) {gtx(g88)} W {g88[2]:.2f}; ref(m4)/OA-VCB(m4) {gtx(g44)}; ref(m4)/ref(m8) {gr[0]:.3f}; ref(m4)/OA-VCB(m8) {gtx(g48)}")
        lines.append(f"{nm} & {fmt(e8.mean())} & {fmt(r8.mean())} & {gtx(g88)} & {g88[2]:.2f} & {gtx(g44)} & {gr[0]:.2f} \\\\")
    write('tab_m8', "Transfers with $m=8$ Jacobi steps (held-out data seed 1, initialisation seed 0): OA-VCB trained with $m=8$ and its classical reference "
          "with $m=8$; mean energy errors, paired $G$ [95\\% CI] and $W$; for comparison the $m=4$ margin and the improvement of the classical reference from "
          "$m=4$ to $m=8$.", 'tab:m8', 'lllllll', 'class & OA-VCB ($m=8$) & reference ($m=8$) & $G$ [95\\% CI] & $W$ & $G$ at $m=4$ & ref.\\ $m=4$/$m=8$', lines)

NUM.close()
open(os.path.join(OUT, 'MANIFEST.txt'), 'w').write("\n".join(sorted(USED)) + "\n")
print('tables written to', OUT, f'({len(USED)} input files, MANIFEST.txt)')

"""generate the development tables of the Route C scoping note from the raw logs / JSON records (no hand transcription)."""
import os, re, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda n: open(os.path.join(HERE, n)).read().splitlines()
out = {}
num = r'([0-9.]+(?:e[+-]?[0-9]+)?)'

# T1 joint vs fixed lifting (global oracles)
rows = ['| Family | Input | Rel. algebraic error | Best flux, fixed lifting | Best flux, lifting optimised (joint) |', '|---|---|---:|---:|---:|']
for l in L('logs_rc_joint_oracle.log'):
    m = re.match(r'(\w+)\s+(.+?)\s+rel err ' + num + r':\s+oracle \(MVY lifting\) eff\s+' + num + r'\s+joint oracle eff\s+' + num, l)
    if m: rows.append(f'| {m[1]} | {m[2]} | {float(m[3]):.2g} | {float(m[4]):.2f} | {float(m[5]):.2f} |')
out['JOINT'] = '\n'.join(rows)

# T2 contrast sweep
rows = ['| Family | Contrast | Input | MVY multilevel (fixed lifting) | Best flux, fixed lifting | G2 lifting-free | OA2+Z lifting-free | Joint optimum |', '|---|---:|---|---:|---:|---:|---:|---:|']
for l in L('logs_rc_mvy_contrast.log'):
    m = re.match(r'(\w+)\s+c=\s*' + num + r'\s*\|(.*)', l)
    if not m: continue
    for part in m[3].split('|'):
        q = re.match(r'\s*(\w+): MVY\s+' + num + r' oracle\(MVY lift\)\s+' + num + r' G2-LF\s+' + num + r' OA2-LF\s+' + num + r' joint\s+' + num, part)
        if q: rows.append(f'| {m[1]} | {float(m[2]):.0e} | {q[1]} | {float(q[2]):.2f} | {float(q[3]):.2f} | {float(q[4]):.2f} | {float(q[5]):.2f} | {float(q[6]):.2f} |')
out['CONTRAST'] = '\n'.join(rows)

# T3 lifting-free vs relaxed at the paper contrast
rows = ['| Family | Input | Joint optimum | G2 relaxed | G2 lifting-free | G1 relaxed | G1 lifting-free | OA2+Z relaxed | OA2+Z lifting-free |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
fam = None
for l in L('logs_rc_test_liftfree.log'):
    m = re.match(r'(\w+) id 300', l)
    if m: fam = m[1]; continue
    m = re.match(r'(.+?)\s+joint\s+' + num + r'\s*\|(.*)', l)
    if not m: continue
    vals = re.findall(r'relaxed\s+' + num + r' lift-free\s+' + num, m[3])
    rows.append(f'| {fam} | {m[1].strip()} | {float(m[2]):.3f} | ' + ' | '.join(f'{float(a):.3f} | {float(b):.3f}' for a, b in vals) + ' |')
out['LIFTFREE'] = '\n'.join(rows)

# T4 stress 1e6
rows = ['| Family | Input | Rel. algebraic error | G2 lifting-free | OA2+Z lifting-free | Joint optimum |', '|---|---|---:|---:|---:|---:|']
for l in L('logs_rc_stress_1e6.log'):
    m = re.match(r'(\w+)\s+c=1e6 nk=\d+ (.+?)\s+rel err ' + num + r': G2-LF\s+' + num + r'\s+OA2-LF\s+' + num + r'\s+joint\s+' + num, l)
    if m: rows.append(f'| {m[1]} | {m[2].strip()} | {float(m[3]):.1g} | {float(m[4]):.2f} | {float(m[5]):.3f} | {float(m[6]):.3f} |')
out['STRESS'] = '\n'.join(rows)

# T5 index sweep
rows = ['| Family | Contrast | G2 floating max (p95) | G2 boundary max | G1 floating max (p95) | G1 boundary max | OA2 floating max (p95) |', '|---|---:|---:|---:|---:|---:|---:|']
for l in L('logs_rc_index_sweep.log'):
    m = re.match(r'(\w+)\s+c=\s*' + num + r' nk=\d+ \| (.*)', l)
    if not m: continue
    v = re.findall(r'floating max\s+' + num + r' p95\s+' + num + r' \| boundary max\s+' + num, m[3])
    (g2, g2p, g2b), (g1, g1p, g1b), (oa, oap, oab) = [tuple(map(float, t)) for t in v]
    rows.append(f'| {m[1]} | {float(m[2]):.0e} | {g2:.2f} ({g2p:.2f}) | {g2b:.2f} | {g1:.2f} ({g1p:.2f}) | {g1b:.2f} | {oa:.2f} ({oap:.2f}) |')
out['INDEX'] = '\n'.join(rows)

# T6 goal certificates
rows = ['| Family | Input pair | Method | D: radius / (e_u e_z) | D: contains J(u_h) | C: radius / abs(J) | C: contains reference bracket |', '|---|---|---|---:|---|---:|---|']
for fam in ('channels', 'barriers'):
    for r in json.load(open(os.path.join(HERE, f'dev_goal_{fam}.json'))):
        eff = r['radD'] / (r['eu'] * r['ez']) if r['eu'] * r['ez'] > 0 else float('nan')
        effs = 'n/a (exact)' if r['eu'] * r['ez'] == 0 else f'{eff:.2f}'
        rows.append(f"| {fam} | {r['pair']} | {r['method']} | {effs} | {'yes' if r['okD'] else 'NO'} | {r['radC'] / abs(r['J5']):.1f} | {'yes' if r['okC'] else 'NO'} |")
out['GOAL'] = '\n'.join(rows)
gl = {fam: json.load(open(os.path.join(HERE, f'dev_goal_{fam}.json'))) for fam in ('channels', 'barriers')}
out['GOAL_CHECKS'] = dict(n=sum(len(v) for v in gl.values()), allD=all(r['okD'] for v in gl.values() for r in v),
                          allC=all(r['okC'] for v in gl.values() for r in v),
                          max_moment_defect=max(r['moment_defect'] for v in gl.values() for r in v),
                          max_div_defect=max(r['div_defect'] for v in gl.values() for r in v),
                          ref={fam: dict(J5=v[0]['J5'], r5=v[0]['r5'], Jh=v[0]['J_h']) for fam, v in gl.items()})
gs = json.load(open(os.path.join(HERE, 'global_stability.json')))
rows = ['| Family | Contrast | C_glob (worst case over all inputs) |', '|---|---:|---:|'] + [f"| {r['family']} | {r['contrast']:.0e} | {r['C_glob']:.3f} |" for r in gs]
out['GLOBAL'] = '\n'.join(rows)
for l in L('logs_rc_disc_size.log'):
    m = re.match(r'(\w+) id 300 L3: eta_total\(u_h\) / \|\|\|u_h\|\|\| = ' + num, l)
    if m: out[f'DISC_{m[1]}'] = float(m[2])
# development maxima quoted in the summary
import math
def parse_contrast():
    res = []
    for l in L('logs_rc_mvy_contrast.log'):
        m = re.match(r'(\w+)\s+c=\s*' + num + r'\s*\|(.*)', l)
        if not m: continue
        for part in m[3].split('|'):
            q = re.match(r'\s*(\w+): MVY\s+' + num + r' oracle\(MVY lift\)\s+' + num + r' G2-LF\s+' + num + r' OA2-LF\s+' + num + r' joint\s+' + num, part)
            if q: res.append((m[1], float(m[2]), q[1], *map(float, q.groups()[1:])))
    return res
cs = parse_contrast()
out['SUMMARY_CHECKS'] = dict(
    oa_over_joint_max_contrast_sweep=max(r[6] / r[7] for r in cs),
    oa_eff_range_contrast_sweep=[min(r[6] for r in cs), max(r[6] for r in cs)],
    mvy_c1=[(r[0], r[2], r[3]) for r in cs if r[1] == 1.0],
    mvy_c1e4=[(r[0], r[2], r[3]) for r in cs if r[1] == 1e4])
json.dump(out, open(os.path.join(HERE, 'scoping_tables.json'), 'w'), indent=1)
for k, v in out.items(): print(f'=== {k}\n{v}\n')

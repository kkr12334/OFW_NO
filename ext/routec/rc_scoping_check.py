"""named claims check for ROUTE_C_SCOPING_2026-10-18.md: every summary number is recomputed from the raw records."""
import os, re, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, 'scoping_tables.json')))
doc = open(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else 'ROUTE_C_SCOPING_2026-10-18.md')).read()
L = lambda n: open(os.path.join(HERE, n)).read()
num = r'([0-9.]+(?:e[+-]?[0-9]+)?)'
checks = []
def check(name, cond, detail=''):
    checks.append((name, bool(cond), detail))
# contrast sweep
cs = []
for l in L('logs_rc_mvy_contrast.log').splitlines():
    m = re.match(r'(\w+)\s+c=\s*' + num + r'\s*\|(.*)', l)
    if not m: continue
    for part in m[3].split('|'):
        q = re.match(r'\s*(\w+): MVY\s+' + num + r' oracle\(MVY lift\)\s+' + num + r' G2-LF\s+' + num + r' OA2-LF\s+' + num + r' joint\s+' + num, part)
        if q: cs.append(dict(fam=m[1], c=float(m[2]), inp=q[1], mvy=float(q[2]), ora=float(q[3]), g2=float(q[4]), oa=float(q[5]), joint=float(q[6])))
g = lambda fam, c, inp, k: [r[k] for r in cs if r['fam'] == fam and r['c'] == c and r['inp'] == inp][0]
check('MVY c=1 channels PCG30 = 1.13', f"{g('channels', 1.0, 'PCG30', 'mvy'):.2f}" == '1.13')
check('MVY c=1 channels u_c = 1.40', f"{g('channels', 1.0, 'u_c', 'mvy'):.2f}" == '1.40')
check('MVY c=1e4 channels PCG30 = 22.05', f"{g('channels', 1e4, 'PCG30', 'mvy'):.2f}" == '22.05')
check('MVY c=1e4 barriers u_c = 4.78', f"{g('barriers', 1e4, 'u_c', 'mvy'):.2f}" == '4.78')
check('OA2-LF effectivity range in contrast sweep within [1.02, 1.25]', 1.015 <= min(r['oa'] for r in cs) and max(r['oa'] for r in cs) <= 1.255, (min(r['oa'] for r in cs), max(r['oa'] for r in cs)))
check('OA2-LF / joint <= 1.03 in contrast sweep', max(r['oa'] / r['joint'] for r in cs) <= 1.03 + 1e-9, max(r['oa'] / r['joint'] for r in cs))
# lift-free log at 1e4
lf = []
for l in L('logs_rc_test_liftfree.log').splitlines():
    m = re.match(r'(.+?)\s+joint\s+' + num + r'\s*\|(.*)', l)
    if m:
        v = re.findall(r'relaxed\s+' + num + r' lift-free\s+' + num, m[3])
        lf.append((m[1].strip(), float(m[2]), float(v[2][1])))
check('OA2+Z lift-free / joint <= 1.03 at 1e4 (all inputs)', max(o / j for _, j, o in lf) <= 1.03 + 1e-9, max(o / j for _, j, o in lf))
check('OA2+Z lift-free non-noise effectivity within [1.02, 1.25] at 1e4', all(1.015 <= o <= 1.255 for n, _, o in lf if 'noise' not in n))
# joint oracle
jo = re.findall(r'(\w+)\s+u_h \+ 10% noise\s+rel err ' + num + r':\s+oracle \(MVY lifting\) eff\s+' + num + r'\s+joint oracle eff\s+' + num, L('logs_rc_joint_oracle.log'))
fixed = sorted(float(t[2]) for t in jo); joint = sorted(float(t[3]) for t in jo)
check('noise, fixed-lifting best flux 15-30-fold', int(fixed[0]) == 15 and round(fixed[-1]) == 30, fixed)
check('noise, joint optimum 1.34-1.38', f'{joint[0]:.2f}' == '1.34' and f'{joint[-1]:.2f}' == '1.38', joint)
# stress
st = re.findall(r'(\w+)\s+c=1e6 nk=\d+ (.+?)\s+rel err ' + num + r': G2-LF\s+' + num + r'\s+OA2-LF\s+' + num + r'\s+joint\s+' + num, L('logs_rc_stress_1e6.log'))
gap = max(float(t[4]) / float(t[5]) for t in st); worst = [t for t in st if float(t[4]) / float(t[5]) == gap][0]
check('1e6: max OA/joint gap 21% at channels PCG100', round(100 * (gap - 1)) == 21 and worst[0] == 'channels' and worst[1].strip() == 'PCG100', (gap, worst[:2]))
# index
ix = {}
for l in L('logs_rc_index_sweep.log').splitlines():
    m = re.match(r'(\w+)\s+c=\s*' + num + r' nk=\d+ \| (.*)', l)
    if m: ix[(m[1], float(m[2]))] = [tuple(map(float, t)) for t in re.findall(r'floating max\s+' + num + r' p95\s+' + num + r' \| boundary max\s+' + num, m[3])]
check('G2 channels floating max about 34 at 1e4, 340 at 1e6', round(ix[('channels', 1e4)][0][0]) == 34 and round(ix[('channels', 1e6)][0][0], -1) == 340)
check('OA barriers floating max 1.27 at every contrast', all(f'{ix[("barriers", c)][2][0]:.2f}' == '1.27' for c in (1, 10, 100, 1e3, 1e4, 1e6)))
check('OA channels floating max 3.6 at 1e4 and 36 at 1e6', f'{ix[("channels", 1e4)][2][0]:.1f}' == '3.6' and round(ix[('channels', 1e6)][2][0]) == 36)
check('geometric index grows like c^1/2 (ratio 1e6/1e4 between 8 and 12 for G2 channels and G1 both)', all(8 <= ix[(f, 1e6)][k][0] / ix[(f, 1e4)][k][0] <= 12 for f, k in (('channels', 0), ('channels', 1), ('barriers', 1))))
# goal certificates
gl = {f: json.load(open(os.path.join(HERE, f'dev_goal_{f}.json'))) for f in ('channels', 'barriers')}
ex = {f: [r for r in v if r['pair'].startswith('exact')][0] for f, v in gl.items()}
check('C radius / |J| exact: 2.6 channels, 18 barriers', f"{ex['channels']['radC'] / abs(ex['channels']['J5']):.1f}" == '2.6' and round(ex['barriers']['radC'] / abs(ex['barriers']['J5'])) == 18)
check('all D intervals contain J(u_h)', all(r['okD'] for v in gl.values() for r in v))
check('all C intervals contain the L5 reference bracket', all(r['okC'] for v in gl.values() for r in v))
nn = {f: [r for r in v if r['pair'].startswith('OA-VCB') and r['method'].startswith('OA2')][0] for f, v in gl.items()}
check('OA-VCB D effectivity 1.30 channels / 1.20 barriers', f"{nn['channels']['radD'] / (nn['channels']['eu'] * nn['channels']['ez']):.2f}" == '1.30' and f"{nn['barriers']['radD'] / (nn['barriers']['eu'] * nn['barriers']['ez']):.2f}" == '1.20')
check('L3 total estimator about 29% of |||u_h||| (channels), 39% (barriers)', round(100 * T['DISC_channels']) == 29 and round(100 * T['DISC_barriers']) == 39, (T.get('DISC_channels'), T.get('DISC_barriers')))
l5 = [float(x) for x in re.findall(r'eta5_u/\|\|\|u5\|\|\| ' + num, L('dev_goal_channels.log') + L('dev_goal_barriers.log'))]
check('L5 relative energy bounds 15-17%', round(100 * min(l5)) == 15 and round(100 * max(l5)) == 17, l5)
check('max moment defect <= 1e-11 relative', T['GOAL_CHECKS']['max_moment_defect'] <= 1e-11, T['GOAL_CHECKS']['max_moment_defect'])
# global stability
gs = json.load(open(os.path.join(HERE, 'global_stability.json')))
check('global stability computed for both families at 4 contrasts', len(gs) == 8)
cg = sorted(r['C_glob'] for r in gs)
check('C_glob range 2.06-2.14', f'{cg[0]:.2f}' == '2.06' and f'{cg[-1]:.2f}' == '2.14', cg)
# facts
fe = L('logs_rc_feasibility.log')
check('non-QM 1.9 (0-7) both families', fe.count('non-quasi-monotone 1.9 (min 0, max 7)') == 2)
fa = L('logs_rc_facts.log') if os.path.exists(os.path.join(HERE, 'logs_rc_facts.log')) else ''
check('facts: near-kernel counts [58 31 4 7], 7424 elements, angles 21.2-126.5, 2048 obtuse',
      '[58 31  4  7]' in fa and 'elements 7424' in fa and 'min angle 21.2 max angle 126.5' in fa and 'obtuse elements 2048' in fa)
check('stiffness equals H.A_fine to 5e-15', 'stiffness matches H.A_fine: 5.0' in L('logs_rc_test_dev.log') if os.path.exists(os.path.join(HERE, 'logs_rc_test_dev.log')) else False)
check('no unfilled placeholder in the note', '{{' not in doc)
fails = [c for c in checks if not c[1]]
for n, ok, d in checks: print(('ok   ' if ok else 'FAIL ') + n + (f'   [{d}]' if d != '' else ''))
print(f'{len(checks)} named checks; failures: {len(fails)}')

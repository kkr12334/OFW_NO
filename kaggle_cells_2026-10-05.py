# Kaggle cells, 5 Oct: the paper runs (PAPER_PLAN_2026-10-05.md).  First upload the CURRENT code to the Kaggle dataset
# (train_oa_layer.py, train_scalar.py, train_elasticity.py, oa_vcycle.py, fe_metrics.py (new) and the rest of the folder);
# no patch cell needed.  Check after the upload:  !grep -c "review, finding B" /kaggle/input/*/oa_vcycle.py  -> 1
# Revised 5 Oct after the independent review (INDEPENDENT_REVIEW_2026-10-05.md):
#   * every concurrent job writes to ITS OWN directory (finding E: two jobs on one directory raced on the pool cache and
#     the results file); a job's sequential commands may share a directory;
#   * --certify 0 on the Darcy runs (the default 1 certified every model at L3: slow, and the bound it printed was not a
#     guaranteed one -- finding A);
#   * the drivers now also report the FE (mass-matrix) L2 error, the number of test samples worse than u_c in energy, the
#     plain-cycle output of oa-vcb / oa-vcq (loss-matched control of the 2 x 2 ablation) and the V-cycle setup time.
# Revised again 5 Oct (review follow-up): deflation / Galerkin output now use the float64 assembled operator and residual
#   (acceptance 2 d^T b - d^T A d >= 0), gradients safe at zero residuals; residual-linear controls oa-vcl / oa-vcbl added
#   (sessions 2 and 3).  RUN SESSION 0 FIRST (~10 min): the precision / gradient regression tests ON THE GPU and the
#   linear-control check; the cell must end with "ALL JOBS PASSED -- PREFLIGHT PASS" (it raises otherwise).  Upload legacy/ and diagnostics_cost/ too.
# Run with Save Version -> Save & Run All (Commit), GPU T4 x2.  One cell per session; each job runs its commands in order.
import subprocess, os, time, glob
CODE = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/train_oa_layer.py", recursive=True))][0]
OUT = "/kaggle/working"; print("code from", CODE)

# common protocol: Galerkin coarse solve, k0 2, L3, 300 epochs, mix loss, 22k parameter budget for EVERY model
DARCY = ("python -u train_oa_layer.py --coarse perforated --coarse-n 9 --field channels --contrast 1e4 --n 400 --epochs 300 "
         "--loss mix --lr 2e-3 --route V --width 32 --layers 4 --coarse-solve galerkin --level 3 --k0 2 --params 22000 "
         "--certify 0 --strict-code 1")
ANISO = ("python -u train_scalar.py --pde aniso --bc insulated --level 3 --contrast 1e2 --k0 2 --n 400 --epochs 300 "
         "--loss mix --params 22000 --strict-code 1")
ELAST = ("python -u train_elasticity.py --level 3 --n 400 --epochs 300 --loss mix --k0 2 --contrast 1e2 --params 22000 "
         "--strict-code 1")
seq = lambda cmds: " && ".join(cmds)

SESSION = 0          # <- set 0 (pre-flight, first), then 1, 2, 3, 4, 5 (and 6 only if the quota allows).  Protocol (review, 5 Oct): DATA seed 0 was the development / selection set, so
                     #    the paper tables use held-out data seeds 1, 2, 3 for Darcy (aniso: oa-vcb was never developed on
                     #    that class, its 3 Oct baselines exist for data seeds 0-2 -> keep 0-2 there, stated in the paper)
EVAL = (1, 2, 3)
if SESSION == 0:     # pre-flight on the paper's device: regression tests of the repaired projection (cuda) + linear control
    jobs = {"preflight": ("0", seq(["python -u mathcheck/check_projection_precision.py --device cuda",
                                    "python -u mathcheck/check_linear_smoother.py"]))}    # log: /kaggle/working/preflight.log
elif SESSION == 1:   # Darcy channels, held-out data seeds, ours + matched baselines (~3.5 h per GPU)
    jobs = {"darcy_ours": ("0", seq(f"{DARCY} --data-seed {d} --models oa-vcb oa-ll-op gnn-op --classical 1 "
                                    f"--out-dir {OUT}/darcy_ours" for d in EVAL)),
            "darcy_base": ("1", seq(f"{DARCY} --data-seed {d} --models gnn transolver-op fno-op wno-op lino-op mgno mgno-op "
                                    f"--out-dir {OUT}/darcy_base" for d in EVAL))}
elif SESSION == 2:   # anisotropic tows (oa-vcb, data seeds 0-2) + Darcy random loads (held-out data seed 1)
    jobs = {"aniso": ("0", seq(f"{ANISO} --data-seed {d} --models oa-vcb --classical 1 --out-dir {OUT}/aniso"
                               for d in (0, 1, 2))),
            "random": ("1", seq([f"{DARCY} --load random --data-seed 1 --models oa-vcb oa-ll-op gnn-op --classical 1 --out-dir {OUT}/random",
                                 # residual-linear control on channels (review point 5): oa-vcz vs oa-vcl (plain output),
                                 # oa-vcb (session 1, same config) vs oa-vcbl (Galerkin output); held-out data seed 1
                                 f"{DARCY} --data-seed 1 --models oa-vcz oa-vcl oa-vcbl --classical 1 --out-dir {OUT}/linear_ctrl"]))}
elif SESSION == 3:   # barriers: the 2 x 2 ablation (deflation x projection) on held-out data seed 1; second geometry; published WNO + MgNO
    BAR = DARCY.replace('--contrast 1e4', '--contrast 1e-4') + " --load no-low --data-seed 1"
    jobs = {"barriers_ablation": ("0", f"{BAR} --models oa-vc oa-vcz oa-vcq oa-vcb oa-vcl oa-vcbl oa-ll-op gnn-op --classical 1 --out-dir {OUT}/barriers"),
            "geometry_wno": ("1", seq([f"{DARCY.replace('--coarse perforated', '--coarse polygon')} --data-seed 1 "
                                       f"--models oa-vcb oa-ll-op gnn-op --classical 1 --out-dir {OUT}/polygon",
                                       f"{DARCY.replace('--params 22000', '--params 80000')} --data-seed 1 --models wno-off mgno-off "
                                       f"--out-dir {OUT}/wno_published"]))}
elif SESSION == 5:   # (Astra, 5 Oct night) rows promised by the plan that had no run: k0 0 and the transfer ablation, held out
    K00 = DARCY.replace('--k0 2', '--k0 0')
    jobs = {"k0_0": ("0", seq(f"{K00} --data-seed {d} --models oa-vcb oa-ll-op gnn-op --out-dir {OUT}/k0_0" for d in EVAL)),
            "geo_transfers": ("1", seq([f"{DARCY} --oa-steps 0 --data-seed {d} --models oa-vcb --classical 1 --out-dir {OUT}/geo_transfers"
                                        for d in EVAL] +
                                       [f"{DARCY.replace('--contrast 1e4', '--contrast 1e-4')} --load no-low --oa-steps 0 --data-seed 1 "
                                        f"--models oa-vcb --classical 1 --out-dir {OUT}/geo_transfers"]))}
elif SESSION == 6:   # OPTIONAL (~8-9 h per GPU): L4 on a held-out data seed.  Without it, the L4 row is the AUDITED historical
                     # result (data seed 0, checkpoint re-evaluated with the repaired code: diagnostics_cost/checkpoint_audit_L4_k03_seed0.log)
    L4 = DARCY.replace('--level 3 --k0 2', '--level 4 --k0 3')
    jobs = {"L4_vcb": ("0", f"{L4} --data-seed 1 --models oa-vcb --classical 1 --vc-ckpt 1 --out-dir {OUT}/L4_vcb"),
            "L4_base": ("1", f"{L4} --data-seed 1 --models oa-ll-op gnn-op --out-dir {OUT}/L4_base")}
elif SESSION == 4:   # elasticity, stiff bands: oa-vcb (~4-5 h with checkpointing) and the -op baselines at 22k
    jobs = {"elast_vcb": ("0", f"{ELAST} --models oa-vcb --classical 1 --vc-ckpt 1 --out-dir {OUT}/elast_vcb"),
            "elast_base": ("1", seq(f"{ELAST} --seed {s} --models oa-ll-op gnn-op --out-dir {OUT}/elast_base" for s in (0, 1, 2)))}
else:
    raise SystemExit(f"SESSION {SESSION}?")

dirs = {}
for name, (gpu, cmd) in jobs.items():                         # finding E: no directory may be written by two jobs at once
    for d in {w.split()[0] for w in cmd.split('--out-dir ')[1:]}:
        assert dirs.setdefault(d, name) == name, f"{d} is used by both {dirs[d]} and {name}"
procs = {}
for name, (gpu, cmd) in jobs.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs[name] = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env,
                                   stdout=open(f"{OUT}/{name}.log", "w"), stderr=subprocess.STDOUT)
while any(p.poll() is None for p in procs.values()):          # progress every 10 min; keeps a committed run alive
    time.sleep(600)
    for name in jobs:
        tail = open(f"{OUT}/{name}.log").read().splitlines()[-1:] or ['']
        print(time.strftime("%H:%M"), name, procs[name].poll(), tail[0].strip()[:150], flush=True)
codes = {n: p.poll() for n, p in procs.items()}
print("finished:", codes)
bad = [n for n, c in codes.items() if c != 0]
if bad:                                                        # 6 Oct (Astra): a failed job must fail the notebook version
    for n in bad: print(f"--- {n}.log (tail) ---\n" + "\n".join(open(f"{OUT}/{n}.log").read().splitlines()[-30:]))
    raise RuntimeError(f"job(s) failed: {bad} -- see {OUT}/<job>.log")
print("ALL JOBS PASSED" + (" -- PREFLIGHT PASS: proceed to Session 1" if SESSION == 0 else ""))

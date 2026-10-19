# Kaggle main cell -- ACCOUNT A (OFW-NO paper runs).  Change only SESSION.  Expected code hash f8653612eb84 (verified on
# the OFW_NO folder, 6 Oct); every session on BOTH accounts must print it (this cell is not part of the hash).
import subprocess, os, time, glob
CODE = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/train_oa_layer.py", recursive=True))][0]
OUT = "/kaggle/working"; print("code from", CODE)
HASH = subprocess.check_output(["python", "-c", "import run_identity as r; print(r.CODE_HASH)"], cwd=CODE, text=True).strip()
print("code hash", HASH); assert HASH == "f8653612eb84", f"code hash {HASH} != f8653612eb84: wrong or modified upload -- stop"
# 8 Oct: the WNO baselines need pytorch_wavelets + PyWavelets, which Kaggle does not preinstall (notebook Internet must be ON).
# Without them wno-off stops with ModuleNotFoundError and wno-op SILENTLY falls back to wavelet_fallback.py.
subprocess.run(["python", "-m", "pip", "install", "-q", "pytorch_wavelets", "PyWavelets"], check=True)
WB = subprocess.check_output(["python", "-c", "import pywt, pytorch_wavelets, ofwno_nets as o; print(o._WAVELET_BACKEND)"],
                             cwd=CODE, text=True).strip().splitlines()[-1]
print("wavelet backend", WB); assert WB == "pytorch_wavelets", "pytorch_wavelets not importable: turn notebook Internet ON and rerun"
DARCY = ("python -u train_oa_layer.py --coarse perforated --coarse-n 9 --field channels --contrast 1e4 --n 400 --epochs 300 "
         "--loss mix --lr 2e-3 --route V --width 32 --layers 4 --coarse-solve galerkin --level 3 --k0 2 --params 22000 "
         "--certify 0 --strict-code 1")
ANISO = ("python -u train_scalar.py --pde aniso --bc insulated --level 3 --contrast 1e2 --k0 2 --n 400 --epochs 300 "
         "--loss mix --params 22000 --strict-code 1")
ELAST = ("python -u train_elasticity.py --level 3 --n 400 --epochs 300 --loss mix --k0 2 --contrast 1e2 --params 22000 "
         "--strict-code 1")
seq = lambda cmds: " && ".join(cmds)
EVAL = (1, 2, 3)
PREFLIGHT = {"preflight": ("0", seq(["python -u mathcheck/check_projection_precision.py --device cuda",
                                     "python -u mathcheck/check_linear_smoother.py"]))}

ACCOUNT = "A (yours)"
SESSION = 0          # <- run in this order: 0, then 1, then 3, then "3b"   (L4 is a separate cell: account_A_L4.py)
if SESSION == 0:
    jobs = PREFLIGHT
elif SESSION == 1:   # Darcy channels, held-out data seeds 1-3: ours + matched baselines (~6 h; GPU 1 is the long job)
    # 6 Oct (independent review): oa-vc added -- the main-table ablation row (no deflation / no Galerkin output) on held-out
    # seeds; it rides on GPU 0, which finishes ~2 h before the baselines on GPU 1, so the session does not get longer
    jobs = {"darcy_ours": ("0", seq(f"{DARCY} --data-seed {d} --models oa-vcb oa-vc oa-ll-op gnn-op --classical 1 "
                                    f"--out-dir {OUT}/darcy_ours" for d in EVAL)),
            "darcy_base": ("1", seq(f"{DARCY} --data-seed {d} --models gnn transolver-op fno-op wno-op lino-op mgno mgno-op "
                                    f"--out-dir {OUT}/darcy_base" for d in EVAL))}
elif SESSION == 3:   # barriers factorial + linear controls; polygon geometry; published WNO / MgNO (~5 h)
    BAR = DARCY.replace('--contrast 1e4', '--contrast 1e-4') + " --load no-low --data-seed 1"
    jobs = {"barriers_ablation": ("0", f"{BAR} --models oa-vc oa-vcz oa-vcq oa-vcb oa-vcl oa-vcbl oa-ll-op gnn-op --classical 1 --out-dir {OUT}/barriers"),
            "geometry_wno": ("1", seq([f"{DARCY.replace('--coarse perforated', '--coarse polygon')} --data-seed 1 "
                                       f"--models oa-vcb oa-ll-op gnn-op --classical 1 --out-dir {OUT}/polygon",
                                       f"{DARCY.replace('--params 22000', '--params 80000')} --data-seed 1 --models wno-off mgno-off "
                                       f"--out-dir {OUT}/wno_published"]))}
elif SESSION == "3b":   # 8 Oct: rerun of the baselines lost to the missing pytorch_wavelets (~1.5 h)
    # GPU 0: the published-size WNO (failed in Session 3 at import).  GPU 1: the published-size MgNO (same command in
    # Session 3, after wno-off), then wno-op on held-out seeds 1-3 with the REAL pytorch_wavelets backend (Session 1's
    # wno-op ran on the fallback DWT: same db4 periodised transform, different implementation -- this row replaces it).
    P80 = DARCY.replace('--params 22000', '--params 80000') + " --data-seed 1"
    jobs = {"wno_published": ("0", f"{P80} --models wno-off --out-dir {OUT}/wno_published"),
            "mgno_wno_op": ("1", seq([f"{P80} --models mgno-off --out-dir {OUT}/mgno_published"] +
                                     [f"{DARCY} --data-seed {d} --models wno-op --out-dir {OUT}/wno_op_real" for d in EVAL]))}
else:
    raise SystemExit(f"SESSION {SESSION} does not belong to this cell (account A main cell runs 0, 1, 3, '3b'; L4 = account_A_L4.py)")

dirs = {}
for name, (gpu, cmd) in jobs.items():
    for d in {w.split()[0] for w in cmd.split('--out-dir ')[1:]}:
        assert dirs.setdefault(d, name) == name, f"{d} is used by both {dirs[d]} and {name}"
procs = {}
for name, (gpu, cmd) in jobs.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs[name] = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env,
                                   stdout=open(f"{OUT}/{name}.log", "w"), stderr=subprocess.STDOUT)
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    for name in jobs:
        tail = open(f"{OUT}/{name}.log").read().splitlines()[-1:] or ['']
        print(time.strftime("%H:%M"), name, procs[name].poll(), tail[0].strip()[:150], flush=True)
codes = {n: p.poll() for n, p in procs.items()}
print("finished:", codes)
bad = [n for n, c in codes.items() if c != 0]
if bad:
    for n in bad: print(f"--- {n}.log (tail) ---\n" + "\n".join(open(f"{OUT}/{n}.log").read().splitlines()[-30:]))
    raise RuntimeError(f"job(s) failed: {bad} -- see {OUT}/<job>.log")
print(f"ALL JOBS PASSED (account {ACCOUNT}, session {SESSION})" + (" -- PREFLIGHT PASS" if SESSION == 0 else ""))

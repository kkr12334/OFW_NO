# Kaggle L4 cell -- ACCOUNT A, run LAST, separately (OFW-NO paper runs; expected code hash f8653612eb84).  Change only SESSION.
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

ACCOUNT = "A (yours), L4"
SESSION = 6          # L4 on held-out data seed 1 (~9 h; within the 12 h session limit)
L4 = DARCY.replace('--level 3 --k0 2', '--level 4 --k0 3')
jobs = {"L4_vcb": ("0", f"{L4} --data-seed 1 --models oa-vcb --classical 1 --vc-ckpt 1 --out-dir {OUT}/L4_vcb"),
        "L4_base": ("1", f"{L4} --data-seed 1 --models oa-ll-op gnn-op --out-dir {OUT}/L4_base")}

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

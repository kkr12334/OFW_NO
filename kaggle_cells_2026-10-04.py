# Kaggle cells, 4 Oct evening.  CELL 1: patch (adds --vc-ckpt to a copy of the code).  Not needed once the Kaggle dataset
# holds the current oa_vcycle.py / train_oa_layer.py (then set CODE to the dataset folder in CELL 2).
import os, shutil, glob
DST = "/kaggle/working/code_ck"
SRC = [os.path.dirname(p) for p in glob.glob("/kaggle/working/code/train_oa_layer.py")
       + sorted(glob.glob("/kaggle/input/**/train_oa_layer.py", recursive=True))][0]
print("code from", SRC)
shutil.rmtree(DST, ignore_errors=True); shutil.copytree(SRC, DST, symlinks=False)
def patch(fn, pairs):
    p = os.path.join(DST, fn); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, f"{fn}: patch anchor not found -- code differs from the expected version"
        s = s.replace(old, new)
    open(p, "w").write(s)
patch("oa_vcycle.py", [
 ("import numpy as np, scipy.sparse as sp, torch, torch.nn as nn\n",
  "import numpy as np, scipy.sparse as sp, torch, torch.nn as nn\nfrom torch.utils.checkpoint import checkpoint\n"),
 ("project=False, deflate=False):\n        super().__init__()",
  "project=False, deflate=False, ckpt=False):\n        super().__init__()\n        self.ckpt = ckpt   # activation checkpointing of the smoothers (same gradients, less memory)"),
 ("    def _cycle(self, l, b, parts=False):\n        j = 0 if self.share else l\n        x = self.pre[j](b, self.lv[l], self._A[l])",
  "    def _smooth(self, S, b, l):\n        if self.ckpt and self.training and torch.is_grad_enabled():\n"
  "            return checkpoint(lambda b_: S(b_, self.lv[l], self._A[l]), b, use_reentrant=False)\n"
  "        return S(b, self.lv[l], self._A[l])\n\n"
  "    def _cycle(self, l, b, parts=False):\n        j = 0 if self.share else l\n        x = self._smooth(self.pre[j], b, l)"),
 ("        xp = self.post[j](b - self._matvec(l, x + xc), self.lv[l], self._A[l])",
  "        xp = self._smooth(self.post[j], b - self._matvec(l, x + xc), l)"),
])
patch("train_oa_layer.py", [
 ("ap.add_argument('--oa-steps', type=int, default=4,",
  "ap.add_argument('--vc-ckpt', type=int, default=0, help='oa-vc*: 1 = activation checkpointing of the smoothers')\n"
  "ap.add_argument('--oa-steps', type=int, default=4,"),
 ("mk = lambda w: OAVCycle(OAD, width=w, nu=4, project=_pj, deflate=bool(_pj))",
  "mk = lambda w: OAVCycle(OAD, width=w, nu=4, project=_pj, deflate=bool(_pj), ckpt=bool(args.vc_ckpt))"),
])
print("patched code in", DST)

# CELL 2: run.  GPU 0: L4 oa-vcb (oa-ll-op L4 is already done, from the log); GPU 1: k0 0 seeds 1 and 2, one after the other.
import subprocess, os, time
CODE, OUT = "/kaggle/working/code_ck", "/kaggle/working"
BASE = ("python -u train_oa_layer.py --coarse perforated --coarse-n 9 --field channels --contrast 1e4 "
        "--n 400 --epochs 300 --loss mix --lr 2e-3 --route V --width 32 --layers 4 --coarse-solve galerkin")
jobs = {
    "L4_vcb":     ("0", BASE + f" --level 4 --k0 3 --models oa-vcb --classical 1 --vc-ckpt 1 --out-dir {OUT}/runs_L4"),
    "k0_0_seeds": ("1", " && ".join(BASE + f" --level 3 --k0 0 --models oa-ll-op oa-vcb --seed {s} --out-dir {OUT}/runs_k0"
                                    for s in (1, 2))),
}
procs = {}
for name, (gpu, cmd) in jobs.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs[name] = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env,
                                   stdout=open(f"{OUT}/{name}.log", "w"), stderr=subprocess.STDOUT)
while any(p.poll() is None for p in procs.values()):          # progress every 10 min; keeps a committed run alive
    time.sleep(600)
    for name in jobs:
        tail = open(f"{OUT}/{name}.log").read().splitlines()[-1:] or ['']
        print(time.strftime("%H:%M"), name, procs[name].poll(), tail[0].strip(), flush=True)
print("finished:", {n: p.poll() for n, p in procs.items()})

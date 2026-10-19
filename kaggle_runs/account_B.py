# Kaggle main cell -- ACCOUNT B (OFW-NO paper runs).  Change only SESSION.  Upload the SAME folder as on account A; expected
# code hash f8653612eb84 (verified on the OFW_NO folder, 6 Oct) in every session (this cell is not part of the hash).
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

ACCOUNT = "B (colleague)"
SESSION = 9          # <- run in this order: 0, then 2, then 7, then 5, then 4 (elasticity last: appendix only); 8 = random loads, seeds 2-3 (11 Oct); 9 = published-Darcy baseline check (11 Oct); 10 = initialisation seeds (11 Oct); 11 = mesh refinement without retraining (11 Oct)
if SESSION == 0:
    jobs = PREFLIGHT
elif SESSION == 2:   # anisotropic tows (data seeds 0-2) + Darcy random loads + channels linear controls (~3 h)
    jobs = {"aniso": ("0", seq(f"{ANISO} --data-seed {d} --models oa-vcb --classical 1 --out-dir {OUT}/aniso"
                               for d in (0, 1, 2))),
            "random": ("1", seq([f"{DARCY} --load random --data-seed 1 --models oa-vcb oa-ll-op gnn-op --classical 1 --out-dir {OUT}/random",
                                 f"{DARCY} --data-seed 1 --models oa-vcz oa-vcl oa-vcbl --classical 1 --out-dir {OUT}/linear_ctrl"]))}
elif SESSION == 4:   # elasticity, stiff bands (~5 h)
    jobs = {"elast_vcb": ("0", f"{ELAST} --models oa-vcb --classical 1 --vc-ckpt 1 --out-dir {OUT}/elast_vcb"),
            "elast_base": ("1", seq(f"{ELAST} --seed {s} --models oa-ll-op gnn-op --out-dir {OUT}/elast_base" for s in (0, 1, 2)))}
elif SESSION == 5:   # k0 0 on held-out seeds 1-3 + geometric-transfer ablation (~4 h)
    K00 = DARCY.replace('--k0 2', '--k0 0')
    jobs = {"k0_0": ("0", seq(f"{K00} --data-seed {d} --models oa-vcb oa-ll-op gnn-op --out-dir {OUT}/k0_0" for d in EVAL)),
            "geo_transfers": ("1", seq([f"{DARCY} --oa-steps 0 --data-seed {d} --models oa-vcb --classical 1 --out-dir {OUT}/geo_transfers"
                                        for d in EVAL] +
                                       [f"{DARCY.replace('--contrast 1e4', '--contrast 1e-4')} --load no-low --oa-steps 0 --data-seed 1 "
                                        f"--models oa-vcb --classical 1 --out-dir {OUT}/geo_transfers"]))}
elif SESSION == 7:   # 6 Oct (independent review: the island mechanism moves into the abstract) -> barriers on THREE held-out
    # data seeds, not one: seeds 2-3 for ours + ablation row + classical (seed 1 is in account A, Session 3), and the
    # standard baselines on seeds 1-3 so the barriers table has the same rows as channels (~5 h)
    BAR = DARCY.replace('--contrast 1e4', '--contrast 1e-4') + " --load no-low"
    BASE = "gnn transolver-op fno-op wno-op lino-op mgno mgno-op".split()
    jobs = {"barriers_s2": ("0", seq([f"{BAR} --data-seed 2 --models oa-vcb oa-vc oa-ll-op gnn-op --classical 1 --out-dir {OUT}/barriers_s2",
                                      f"{BAR} --data-seed 2 --models {' '.join(BASE)} --out-dir {OUT}/barriers_s2",
                                      f"{BAR} --data-seed 1 --models {' '.join(BASE[:4])} --out-dir {OUT}/barriers_base_s1a"])),
            "barriers_s3": ("1", seq([f"{BAR} --data-seed 3 --models oa-vcb oa-vc oa-ll-op gnn-op --classical 1 --out-dir {OUT}/barriers_s3",
                                      f"{BAR} --data-seed 3 --models {' '.join(BASE)} --out-dir {OUT}/barriers_s3",
                                      f"{BAR} --data-seed 1 --models {' '.join(BASE[4:])} --out-dir {OUT}/barriers_base_s1b"]))}
elif SESSION == 8:   # 11 Oct (RANDOM_LOADS_PLAN_2026-10-11.md): random-load OA-VCB on held-out data seeds 2 and 3 (~45 min, both GPUs)
    # same command as Session 2's random job (seed 1), OA-VCB only, one data seed per GPU, separate output folders
    jobs = {f"random_s{d}": (str(d - 2), f"{DARCY} --load random --data-seed {d} --models oa-vcb --classical 1 --out-dir {OUT}/random_s{d}")
            for d in (2, 3)}
elif SESSION == 9:   # 11 Oct: baseline sanity check on the PUBLISHED FNO Darcy benchmark (kaggle_runs/published_darcy.py, ~4-6 h)
    # needs piececonst_r421_N1024_smooth1.mat and _smooth2.mat attached: Add Input -> Datasets -> scaomath/pde-dataset (public)
    PD = "python -u kaggle_runs/published_darcy.py"
    # 11 Oct rerun: load the 1.7 GB .mat files ONCE (cache /kaggle/working/pd_data_85.npz); stops here at once if they are missing
    subprocess.run(f"{PD} prepare {OUT}/pd_prepare", shell=True, cwd=CODE, check=True)
    jobs = {"pd_fno_wno": ("0", seq([f"{PD} fno {OUT}/pd_fno", f"{PD} fno-official {OUT}/pd_fno_official", f"{PD} wno-db4 {OUT}/pd_wno_db4",
                                      f"{PD} wno-db6 {OUT}/pd_wno_db6"])),
            "pd_mgno_tr": ("1", seq([f"{PD} mgno {OUT}/pd_mgno", f"{PD} wno-light {OUT}/pd_wno_light", f"{PD} transolver {OUT}/pd_transolver"]))}
elif SESSION == 10:  # 11 Oct (independent reviews DeepSeek / Gemini): INDEPENDENT INITIALISATION SEEDS 1 and 2 (published runs
    # use --seed 0) for OA-VCB on channels and barriers, held-out data seeds 1-3, plus the residual-linear control OA-VCBL on
    # channels data seed 1 (the "non-linearity" factor).  Classical comparators are deterministic and already recorded
    # (--classical 0).  Same data pools, same protocol; seed is part of the run config, so records do not collide.
    # Last job per GPU: OA-VCB with m = 8 transfer Jacobi steps (seed 0, data seed 1, with its classical members), because the
    # classical reference improves by 7 % (channels) / 30 % (barriers) from m = 4 to 8 (diagnostics_cost/
    # sensitivity_detector_transfers.py): does the learned margin persist with the stronger hierarchy?  (~4.5 h in total)
    BAR = DARCY.replace('--contrast 1e4', '--contrast 1e-4') + " --load no-low"
    jobs = {"init_channels": ("0", seq([f"{DARCY} --data-seed {d} --seed {s} --models oa-vcb --out-dir {OUT}/init_channels"
                                        for d in EVAL for s in (1, 2)] +
                                       [f"{DARCY} --data-seed 1 --seed 1 --models oa-vcbl --out-dir {OUT}/init_channels",
                                        f"{DARCY} --oa-steps 8 --data-seed 1 --models oa-vcb --classical 1 --out-dir {OUT}/m8_channels"])),
            "init_barriers": ("1", seq([f"{BAR} --data-seed {d} --seed {s} --models oa-vcb --out-dir {OUT}/init_barriers"
                                        for d in EVAL for s in (1, 2)] +
                                       [f"{DARCY} --data-seed 1 --seed 2 --models oa-vcbl --out-dir {OUT}/init_barriers_ctrl",
                                        f"{BAR} --oa-steps 8 --data-seed 1 --models oa-vcb --classical 1 --out-dir {OUT}/m8_barriers"]))}
elif SESSION == 11:  # 11 Oct ("toy problem" concern): the L4-trained OA-VCB (14,398 unknowns) applied WITHOUT RETRAINING to the same
    # problems on refined meshes: L5 (58,494 unknowns, all 100 test samples) and L6 (~234k unknowns, 24 test samples), with the
    # classical reference on the same hierarchy.  CPU only (GPU idle); L5 ~3 GB, L6 >6 GB RAM per sample (OOM on a 7 GB machine;
    # Kaggle has ~30 GB).  Local preview (8 samples):
    # L5 G 8.86, W 1.00.  Needs the L4 checkpoint ckpt_b68267e41f_oa-vcb.pt (folder "Results_kaggle_Session L4") in the
    # uploaded folder or attached as a dataset.  (~3 h)
    hits = sorted(glob.glob("/kaggle/input/**/ckpt_b68267e41f_oa-vcb.pt", recursive=True))
    assert hits, "ckpt_b68267e41f_oa-vcb.pt not found: include folder 'Results_kaggle_Session L4' in the upload or attach it"
    ZS = f'ZS_THREADS=2 ZS_OUT={OUT}/zero_shot_scale.json python -u diagnostics_cost/zero_shot_scale.py "{hits[0]}"'
    jobs = {"zs_L5": ("0", f"ZS_BS=2 {ZS.replace('zero_shot_scale.json', 'zs_L5.json')} 5 4 1 1e4 all 100 zs_L4toL5_k4"),
            "zs_L6": ("1", f"ZS_BS=1 {ZS.replace('zero_shot_scale.json', 'zs_L6.json')} 6 5 1 1e4 all 24 zs_L4toL6_k5")}
else:
    raise SystemExit(f"SESSION {SESSION} does not belong to account B (B runs 0, 2, 7, 5, 4, 8, 9, 10, 11)")

dirs = {}
for name, (gpu, cmd) in jobs.items():
    for d in {w.split()[0] for w in cmd.split('--out-dir ')[1:]}:
        assert dirs.setdefault(d, name) == name, f"{d} is used by both {dirs[d]} and {name}"
procs = {}
for name, (gpu, cmd) in jobs.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs[name] = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env,
                                   stdout=open(f"{OUT}/{name}.log", "w"), stderr=subprocess.STDOUT)
k = 0
while any(p.poll() is None for p in procs.values()):
    k += 1; time.sleep(60 if k <= 5 else 600)          # 11 Oct: report the first minutes quickly (early failures), then every 10 min
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

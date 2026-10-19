# OA-VCB: release for "Operator-adapted learned multigrid cycles for high-contrast elliptic problems: accuracy, failure modes and cost"

This repository is the code and data release of the paper: model, training and data-generation code, the raw evaluation
records and checkpoints, the diagnostics and numerical checks cited in the text, the manuscript sources and table
generators, and the pre-specified Stage 1 confirmatory study (protocol, amendments, deviation disclosures, scripts and
records, under `ext/`). `RELEASE_NOTES.md` lists what is included and what is deliberately left out, and why.

## Layout

| path | content |
|---|---|
| top level `*.py` | model, training and data-generation code (49 files; hashed together as `f8653612eb84`) |
| `oavcb/` | the `oavcb` package used by the Stage 1 scripts |
| `diagnostics_cost/`, `diagnostics_elasticity/`, `diagnostics_allen_cahn/`, `mathcheck/` | CPU diagnostics and numerical checks, each with its log |
| `kaggle_runs/` | the Kaggle drivers that produced the stored records |
| `Results_*`, `results_kaggle session 8/`, `resultskaggle_2026-10-06session_1/` | raw records, run logs and checkpoints of the Kaggle sessions |
| `manuscript/` | paper sources, table and figure generators (`make_tables.py`, `make_tables_stage1.py`) |
| `ext/` | Stage 1 (pre-registered confirmatory study): `STAGE1_PROTOCOL_*`, amendments, pre-confirmation note, reports and closure, `stage1_*.py`, frozen inputs, `*_SHA256.txt` freeze files, `runs/` (records), `dev/`, `explore/` |
| `ext/routec/` | Route C pilot: scoping, amendment, deviation disclosures, report revisions, scripts and `runs/` |
| `tools/` | `release_manifest.py` (kept out of the top level because `run_identity.py` hashes top-level `*.py`) |

The directory is named `ext/` (not `stage1/`) because the frozen SHA-256 files, the scripts and the documents refer to
files by their `ext/...` paths; renaming it would break those references.

## What identifies what

| identifier | covers |
|---|---|
| `f8653612eb84` (`run_identity.py`) | MD5 prefix of the **top-level** `*.py` files (model, training, data generation). Stored with every training/evaluation record and checkpoint. Adding, removing or editing any top-level `.py` file changes it, so new helper scripts live in `tools/`. |
| `RELEASE_SHA256.txt` (`tools/release_manifest.py`) | SHA-256 of **every** released file (top-level code, `diagnostics_cost/`, `diagnostics_elasticity/`, `mathcheck/`, `kaggle_runs/`, all `Results_*` folders and checkpoints, `manuscript/` sources and generators), plus one combined digest. Generate it on the exact tree that is committed and archived. |
| `manuscript/tables/MANIFEST.txt` | the 91 input files that `make_tables.py` actually read, as paths relative to this folder. |
| `manuscript/tables/numbers.txt` | every number quoted in the text, with the record it came from. |

## Requirements

`requirements.txt`: Python 3.13, NumPy 2.5, SciPy 1.18, PyTorch 2.14 (CPU is enough for tables, figures and the CPU
diagnostics), PyAMG 5.3, Matplotlib 3.11. Training the WNO baselines additionally needs `pytorch_wavelets` and `PyWavelets`
(installed by the Kaggle drivers). The manuscript builds with a TeX Live 2025/2026 `latexmk`; `elsarticle.cls` and
`elsarticle-num.bst` are included.

## Reproduce the tables, figures and PDF from the stored records

Run from this folder (`OFW_NO/`):

```
python -I manuscript/make_tables.py             # all tables -> manuscript/tables/*.tex, numbers.txt, MANIFEST.txt (about 30 s)
python -I manuscript/figures/make_figures.py    # Figures 2-5 (setup, errormaps, results, avt) -> manuscript/figures/*.pdf
cd manuscript && latexmk -pdf main.tex          # Elsevier build; check.tex is the same content in the article class
cd .. && python -I tools/release_manifest.py    # RELEASE_SHA256.txt for the release
python -I tools/release_manifest.py --check     # verify a downloaded copy
```

`make_tables.py` stops with an error if a record is missing, ambiguous, has unexpected sample ids or does not reproduce
its stored mean; it never fills a gap silently. Figure 1 (`figures/fig_schematic.tex`) is TikZ source.

## Re-run the experiments

* Training and the stored evaluation records: `kaggle_runs/account_A.py`, `account_A_L4.py`, `account_B.py` (Kaggle main cells;
  each session asserts the code hash `f8653612eb84` before running) and `kaggle_runs/published_darcy.py` (Appendix C). The
  drivers call `train_oa_layer.py`, `train_scalar.py` and `train_elasticity.py` with the arguments listed in the cells. Raw
  outputs are in `Results_kaggle_session */`, `results_kaggle session 8/`, `resultskaggle_2026-10-06session_1/` and
  `Results_kaggle_Session L4/` (records `*.jsonl` / `*.json`, run logs, checkpoints `*.pt`).
* CPU diagnostics used in the paper: `diagnostics_cost/` (cost tables `timing_one_platform.py`; stronger smoothers
  `accuracy_vs_time.py`; contained cycles `fixed_cheb_member.py`, `jacobi_member.py`; sensitivity
  `sensitivity_detector_transfers.py`, `detector_theta_learned.py`; harmonic study `harmonic_study.py` and companions; loads
  `rhs_generalisation_s1.py`, `rhs_random_model.py`; grid bridge `grid_floor_exact.py`), each with its log and JSON output.
* Numerical checks of the propositions: `mathcheck/` (`check_poly_containment.py`, `check_projection_precision.py`,
  `check_oa_transpose_pair.py`, `check_oa_vcb_elasticity.py`, `check_oa_vcb_tensor.py`, `check_oa_vcp.py`), each with its log.
* Refinement without retraining (Session 11): `Results_kaggle_session 11/zs_L5.json` (per-sample record, L5),
  `zs_L5.log`, `zs_L6.log` and the earlier partial log copy (`session11_log_partial_2026-10-12_1025.txt`). The L6 run hit the
  platform's session limit after 21 of 24 samples and wrote no per-sample record; `make_tables.py` reads its single-sample log
  lines.

## Stage 1 (`ext/`)

Stage 1 is the pre-specified confirmatory study the revised paper relies on. Read in this order: `ext/STAGE1_PROTOCOL_2026-10-16.md`,
`ext/STAGE1_PROTOCOL_AMENDMENT_1_2026-10-16.md`, `ext/STAGE1_PROTOCOL_AMENDMENT_2_2026-10-18.md`,
`ext/STAGE1_PRECONFIRMATION_NOTE_2026-10-17.md`, the report and its revisions (`STAGE1_REPORT_*`), `ext/STAGE1_CLOSURE_2026-10-18.md`.
The `ext/*_SHA256.txt` files record the hashes of the frozen documents and scripts at the time of each freeze; deviations from
the protocol are disclosed in the amendments and, for the Route C pilot, in `ext/routec/ROUTE_C_PILOT_DEVIATION_*`.
The scripts are run from this folder, e.g. `python -I ext/stage1_tables.py`, and write into `ext/runs/`.

## Not part of the release

Project notes and review correspondence, `legacy/`, `tmp/`, `feasibility/` and `results/` (earlier development runs) are
not included. They are not inputs to any table or figure (see `manuscript/tables/MANIFEST.txt`).
`manuscript/tables/tab_jacobi.tex` was an obsolete table, no longer generated or included, and is omitted.

## Licence and citation

Code (`*.py`, `*.sh` and the generators under `manuscript/`): MIT (`LICENSE`). Data and records (raw evaluation records, logs, checkpoints, Stage 1 records): CC BY 4.0
(`LICENSE-DATA.md`). The manuscript text and figures are not covered by either licence (author copyright; the journal's
terms apply on publication). Cite the paper and this release: Zenodo DOI [10.5281/zenodo.23477955](https://doi.org/10.5281/zenodo.23477955)
(version v1.0.1; the concept DOI for all versions is [10.5281/zenodo.23477954](https://doi.org/10.5281/zenodo.23477954)). The
archived v1.0.1 tree is the one described by `RELEASE_SHA256.txt`. This README note and `CITATION.cff` were edited after
the tag to add the DOI, so `python -I tools/release_manifest.py --check` on the current `main` reports exactly these two files
as changed; run the check on the tagged release (`git checkout v1.0.1`) or on the Zenodo archive to verify the full tree.

# Release notes

Prepared 19 October 2026 from the working folder `OFW_NO/` as an export copy; nothing was deleted or edited in the working
folder. No top-level `*.py` file was changed, so the code hash is still `f8653612eb84` (49 files, verified on the export).
Provenance of the whole tree: `RELEASE_SHA256.txt` (`python -I tools/release_manifest.py --check`).

## Included, and why

| path | reason |
|---|---|
| all 49 top-level `*.py` (including `kaggle_cells_2026-10-04.py`, `kaggle_cells_2026-10-05.py`) | `run_identity.py` hashes every top-level `*.py`; dropping one would change `f8653612eb84`, which is stored with every record |
| `requirements.txt`, `README.md` (from `README_RELEASE.md`) | environment and entry point |
| `diagnostics_cost/`, `diagnostics_elasticity/`, `mathcheck/`, `kaggle_runs/` | named in the release README; inputs of tables and checks |
| `diagnostics_allen_cahn/` | not in the original list; small (4 files), checks for `train_allen_cahn.py` |
| `oavcb/` | not in the original list; imported by `ext/stage1_report_check.py`, `stage1_tables.py`, `stage1_timing*.py` and `train_allen_cahn.py`, so the Stage 1 scripts cannot run without it |
| `Results_*`, `results_kaggle session 8`, `resultskaggle_2026-10-06session_1` (records, logs, checkpoints) | all 91 inputs of `manuscript/tables/MANIFEST.txt` are inside; checkpoints are small (133 files, 18 MB in total) |
| `manuscript/` sources and generators | paper and table generators; `tab_jacobi.tex` omitted (obsolete) |
| `manuscript/make_tables_stage1.py` (from `paper1_v2_2026-10-19/`) | its docstring says it runs as `manuscript/make_tables_stage1.py`; it recomputes the Stage 1 tables from `ext/runs/` |
| `tools/release_manifest.py` | provenance tool |
| `ext/stage1_*.py`, `ext/stage0_repro.py` and `.json`, `frozen_scales.json`, `checkpoint_map.json`, `stage1_preconfirmation.json`, `stage1_preconf_manifest.json` | Stage 1 code and frozen inputs |
| `ext/*_SHA256.txt` | freeze records for the protocol, amendments, pre-confirmation note and report revisions |
| `ext/runs/`, `ext/dev/`, `ext/explore/` | Stage 1 records. `ext/dev/` files are listed in `STAGE1_PROTOCOL_FREEZE_SHA256.txt` and `STAGE1_AMENDMENT_1_SHA256.txt`; `stage1_report_check.py` reads `ext/explore/` |
| `ext/STAGE1_PROTOCOL_2026-10-16.md`, `..._AMENDMENT_1_...`, `..._AMENDMENT_2_...`, `STAGE1_PRECONFIRMATION_NOTE_2026-10-17.md` | frozen protocol, amendments, pre-confirmation note |
| `ext/STAGE1_REPORT_2026-10-17.md`, `STAGE1_REPORT_REV1_2026-10-18.md`, `STAGE1_REPORT_REV1_1_2026-10-18.md`, `STAGE1_CLOSURE_2026-10-18.md` | report revisions and closure; the first two are hashed in `STAGE1_REPORT_SHA256.txt` / `STAGE1_REPORT_REV1_SHA256.txt` |
| `ext/STAGE0_GATE_2026-10-16.md` | gate document for the Stage 0 reproduction (`stage0_repro.py`) |
| `ext/CLAUDE_RESPONSE_TO_ASTRA_STAGE1_REPORT_REVIEW_2026-10-18.md` | exception to "correspondence stays out": it is hashed in `STAGE1_REPORT_REV1_SHA256.txt`, so the freeze record cannot be verified without it |
| `ext/routec/` (scripts, `runs/`, `*_SHA256.txt`, scoping, pilot amendment, deviations 1 and 2 with addendum, pilot reports and revisions, templates, plan) | the deviation disclosures the revised paper relies on live here; not named in the original include list |

## Left out, and why

| path | reason |
|---|---|
| `legacy/`, `tmp/`, `feasibility/`, `results/`, `__pycache__/` | development history or build products; none is an input to a table or figure |
| old `README.md`, `README_oa_lifting.md` (227 KB), `manuscript/tables/tab_jacobi.tex` | superseded / obsolete |
| top-level dated `*.md` notes and review letters, `reviews_2026-10-06/`, `publication-readiness-review.md`, `REVIEW_paper_plan_2026-10-05.txt` | internal planning and correspondence |
| `ext/ASTRA_*`, `ext/CLAUDE_*` (except the one above), `ext/DRAWING_BOARD_*`, `ext/PAPER1_REFRAME_PLAN`, `ext/PROJECT_STOCKTAKE_AND_TODO`, `ext/REVIEW_AUDIT`, `ext/STAGE1_STATUS`, `ext/astra_*` folders | review letters, briefs, memos, planning |
| `ext/drawing_board/`, `ext/superseded_rev1/` | planning material and superseded scripts |
| `ext/routec/ASTRA_*`, `ext/routec/CLAUDE_RESPONSE_*`, `ext/routec/astra_*` folders | review correspondence |
| `diagnostics_cost/astra_*` (6 review-audit scripts and 9 page images, named after the reviewer) | review-response checks, not inputs to any table or figure (none is listed in `manuscript/tables/MANIFEST.txt`, nothing else refers to them) |
| `paper1_v2_2026-10-19/` (other than `make_tables_stage1.py`) | working drafts of the revised manuscript: PDFs, cover letter, handover notes, source archive; these belong in the manuscript submission, not the code release |

## Checks on the export

* Secrets: no Kaggle, GitHub or Hugging Face tokens, passwords or `kaggle.json`; no notebooks.
* Local paths: no `C:\Users\...` or OneDrive paths in any released file.
* Email: the author address appears only in the manuscript and cover-letter sources (approved corresponding-author address).
* **Sanitised copies.** Kaggle account names appeared in four run logs: `Results_kaggle_session 3-3b/{geometry_wno,session3_notebook,session3b_notebook}.log`
  and `Results_kaggle_session 4/elast_base.log` (the author's account and a dataset owner's account). In these four files, and only
  there, each account name is replaced by `<kaggle-user>` (941 replacements, nothing else changed). The released files therefore
  differ from the original logs in the author's working folder, which are kept there unchanged and are not in this repository.
  None of the four is an input of `make_tables.py` or listed in `manuscript/tables/MANIFEST.txt`.
* The reviewer name "Astra" still appears in comments of top-level `*.py` files (these cannot be edited without changing the code
  hash), in comments of some diagnostics scripts, and in the text and one filename of the Stage 1 documents
  (`ext/CLAUDE_RESPONSE_TO_ASTRA_STAGE1_REPORT_REVIEW_2026-10-18.md`, kept because a frozen hash covers it).

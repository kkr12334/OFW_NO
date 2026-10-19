"""Executed-source manifest for the pre-confirmation note (amendment 1, A5.6 / A7.3).  Imports every Stage 1 module (no
data generated), then hashes: ext/*.py, every loaded module under the release root, the frozen JSON inputs, and every
checkpoint the confirmation runners will load.  usage: python -I ext/stage1_freeze.py -> ext/stage1_preconf_manifest.json"""
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
import s1_common, stage1_core, stage1_models, stage1_labels, stage1_stats, stage1_loads, stage1_manifest  # noqa: F401
from stage1_manifest import manifest, sha
CK = json.load(open(os.path.join(HERE, 'checkpoint_map.json')))
used = sorted({os.path.join(ROOT, r['path']) for r in CK if r['level'] == 3 and r['k0'] == 2 and r['oa_steps'] == 4 and r['coarse'] == 'perforated'
               and r['model'] in ('oa-vcb', 'oa-vcbl', 'oa-vcl') and (r['data_seed'] == 1 or r['model'] == 'oa-vcb')})
M = manifest(used)
M['ext_sources'] = {os.path.relpath(p, ROOT): sha(p) for p in sorted(glob.glob(os.path.join(HERE, '*.py')))}
M['frozen_inputs'] = {os.path.relpath(p, ROOT): sha(p) for p in [os.path.join(HERE, f) for f in
                      ('stage1_preconfirmation.json', 'frozen_scales.json', 'checkpoint_map.json', 'dev/stage1_fitted_methods.json', 'dev/stage1_fit.json',
                       'STAGE1_PROTOCOL_2026-10-16.md', 'STAGE1_PROTOCOL_AMENDMENT_1_2026-10-16.md')]}
M.pop('blas'); M['sources'].pop(os.path.relpath(os.path.abspath(sys.argv[0]), ROOT), None)
json.dump(M, open(os.path.join(HERE, 'stage1_preconf_manifest.json'), 'w'), indent=1)
print(json.dumps({k: M[k] for k in ('code_hash', 'versions', 'threads')}, indent=1)); print(len(M['ext_sources']), 'ext sources;', len(M['checkpoints']), 'checkpoints; all match map:', all(v['matches_map'] for v in M['checkpoints'].values()))

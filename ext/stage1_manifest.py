"""Run manifest and executable gates (amendment 1, A7.3).
manifest(ckpts) -> dict: SHA-256 of every executed source under the release root (ext/*.py, diagnostics_cost helpers,
top-level modules: every loaded module whose file lies under ROOT, plus the running script), top-level code hash,
package versions, thread and BLAS settings, checkpoint SHA-256s checked against ext/checkpoint_map.json.
Gate(...) collects named gates; finish() writes them and exits non-zero if any failed."""
import os, sys, io, json, hashlib, platform, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FROZEN_FIT_SHA = '9dffe18716911af29afd32103c772ae30978b4e6adbc2bd3cc511a7c446fa2fd'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def fitted_methods_sha(path):
    return hashlib.sha256(json.dumps(json.load(open(path)), sort_keys=True).encode()).hexdigest()


def manifest(ckpts=()):
    import numpy, scipy, torch, pyamg
    try:
        import matplotlib; mpl = matplotlib.__version__
    except Exception: mpl = None
    import run_identity  # noqa: F401  (so its source is hashed below)
    files = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if not (isinstance(f, str) and os.path.isabs(f)): continue          # skip modules with relative/virtual file names
        if f.startswith(ROOT + os.sep) and f.endswith('.py') and os.path.isfile(f): files[os.path.relpath(f, ROOT)] = sha(f)
    main = os.path.abspath(sys.argv[0]); files[os.path.relpath(main, ROOT)] = sha(main)
    from run_identity import CODE_HASH
    cmap = {r['path']: r['sha256'] for r in json.load(open(os.path.join(HERE, 'checkpoint_map.json')))}
    ck = {}
    for p in ckpts:
        rel = os.path.relpath(p, ROOT); h = sha(p); ck[rel] = dict(sha256=h, matches_map=(cmap.get(rel) == h))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): numpy.show_config()
    return dict(sources=dict(sorted(files.items())), code_hash=CODE_HASH,
                versions=dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                              torch=torch.__version__, pyamg=pyamg.__version__, matplotlib=mpl),
                threads=dict(torch=torch.get_num_threads(), **{v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
                blas=dict(numpy=buf.getvalue(), torch=torch.__config__.show()),
                cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'),
                checkpoints=ck)


class Gates:
    def __init__(self): self.g = {}
    def check(self, name, ok, detail=None):
        self.g[name] = dict(passed=bool(ok), detail=detail); print(f"GATE {'pass' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail is not None else ''), flush=True)
        return ok
    def failed(self): return [k for k, v in self.g.items() if not v['passed']]

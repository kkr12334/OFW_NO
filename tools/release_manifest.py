"""Release manifest (13 Oct 2026, FINAL_SUBMISSION_REVIEW_ASTRA item A).

run_identity.py hashes only the top-level *.py files (the model / training code; hash f8653612eb84 stored with every record).
This script identifies the COMPLETE release: it writes RELEASE_SHA256.txt with the SHA-256 of every released file (code,
diagnostics, mathcheck, Kaggle drivers, raw records, checkpoints, manuscript sources and generators) and one combined digest
over that list.  Run it from the OFW_NO folder on the exact tree that is committed / archived:

    python tools/release_manifest.py            # writes RELEASE_SHA256.txt
    python tools/release_manifest.py --check    # verifies an existing RELEASE_SHA256.txt against the tree
"""
import hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> OFW_NO (kept out of the top level: run_identity.py hashes top-level *.py)
OUT = os.path.join(ROOT, 'RELEASE_SHA256.txt')
SKIP_DIRS = {'__pycache__', '.git', 'tmp', 'legacy', 'feasibility', '.ipynb_checkpoints'}
SKIP_EXT = {'.pyc', '.aux', '.log~', '.fls', '.fdb_latexmk', '.out', '.bbl', '.blg', '.synctex.gz', '.toc'}
SKIP_FILES = {'RELEASE_SHA256.txt'}


def files():
    for d, sub, fs in os.walk(ROOT):
        sub[:] = sorted(s for s in sub if s not in SKIP_DIRS)
        for f in sorted(fs):
            if f in SKIP_FILES or any(f.endswith(e) for e in SKIP_EXT):
                continue
            # LaTeX build logs inside manuscript/ are build products, not inputs
            rel = os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, '/')
            if rel.startswith('manuscript/') and f.endswith('.log'):
                continue
            yield rel


def sha(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = [f"{sha(p)}  {p}" for p in files()]
    body = "\n".join(rows) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    if '--check' in sys.argv:
        old = open(OUT).read().split('\n# combined')[0] + "\n" if os.path.exists(OUT) else ''
        old_rows = [l for l in old.splitlines() if l and not l.startswith('#')]
        new_rows = body.splitlines()
        if old_rows == new_rows:
            print(f'OK: {len(new_rows)} files match RELEASE_SHA256.txt (combined {digest[:16]})')
            return
        a, b = set(old_rows), set(new_rows)
        for l in sorted(a - b): print('listed but changed or missing:', l.split('  ', 1)[1])
        for l in sorted(b - a): print('new or changed:', l.split('  ', 1)[1])
        sys.exit(1)
    header = ("# SHA-256 of every file of the release, relative to the OFW_NO folder (tools/release_manifest.py).\n"
              "# Model / training code hash stored with every record (run_identity.py, top-level *.py only): f8653612eb84\n")
    open(OUT, 'w').write(header + body + f"# combined sha256 of the list above: {digest}\n")
    print(f'RELEASE_SHA256.txt: {len(rows)} files, combined {digest}')


if __name__ == '__main__':
    main()

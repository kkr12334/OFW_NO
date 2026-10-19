"""5 Oct (Astra, point 5): one code-version hash for every driver -- md5 of the top-level *.py files of the code folder.
Recorded with every results line and checkpoint; with --strict-code 1 a driver reuses a finished result or checkpoint only
if it was produced by the SAME code version (otherwise it retrains and says so).  Older records without a hash count as
a different version."""
import glob, hashlib, os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_HASH = hashlib.md5(b''.join(open(f, 'rb').read() for f in sorted(glob.glob(os.path.join(CODE_DIR, '*.py'))))).hexdigest()[:12]


def reusable(record_code, strict, what=''):
    """True if a stored item may be reused; prints why not when strict mode rejects it."""
    if not strict or record_code == CODE_HASH:
        return True
    print(f"  --strict-code: {what} was produced by code version {record_code or 'unknown (pre-hash)'}, "
          f"current {CODE_HASH} -> not reused, recomputed", flush=True)
    return False

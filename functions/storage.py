"""storage.py -- small file helpers for the result folders.

digest(path)       SHA-256 of a file (to tie an SNR curve to the exact coefficients)
write_json(path)   atomic JSON write: a crash never leaves half a file
exclusive(path)    lock, so two processes never train or generate the same thing
"""
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@contextmanager
def exclusive(path):
    """Hold an exclusive lock on `path`; fail at once if another process holds it
    (e.g. the same notebook running in a second kernel)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'{path} is held by another process: '
                               'something is already computing this') from e
        f.seek(0)
        f.truncate()
        f.write(f'{os.getpid()}\n')
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

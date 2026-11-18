import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def config(demo=False):
    return dict(demo=demo, seed=2411, base=4 if demo else 32, first=32 if demo else 256,
                refine=48 if demo else 384, epochs=1 if demo else 50, effective=8,
                threshold=0.5, bootstrap=2000, budget=0.2, lr=0.001)


def stable(value):
    return int(digest(str(value).encode())[:15], 16)


def patients(rows):
    return sorted({row["patient"] for row in rows})


def mean(values):
    return float(np.mean(values)) if len(values) else None

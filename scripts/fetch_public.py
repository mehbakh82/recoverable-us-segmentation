"""Stage pinned public datasets locally; no training, scoring or redistribution."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

ASSETS = (
    ("BUSBRA.zip", "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1", "ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff"),
    ("breast.zip", "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip", "c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355"),
    ("breast.xlsx", "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx", "89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf"),
    ("uclm.zip", "https://data.mendeley.com/public-api/zip/7fvgj4jsp7/download/3", "7cb15270f1a2c920b4cd89122acc5dd399eae0fe545893a44596676225ac313e"),
)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def stage(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, url, expected in ASSETS:
        archive = root/name
        if not archive.exists():
            temp = archive.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as stream:
                shutil.copyfileobj(response,stream)
            if checksum(temp) != expected:
                raise ValueError(f"Release bytes changed: {name}; verify source before changing the protocol")
            os.replace(temp,archive)
        if checksum(archive) != expected:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if not (root/member.filename).resolve().is_relative_to(root):
                        raise ValueError("Invalid ZIP member path")
                bundle.extractall(root)
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    stage(parser.parse_args().root)

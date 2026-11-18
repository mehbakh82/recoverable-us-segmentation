"""Private-data-free source archive and manifest receipt, created before scoring the pilot."""
import argparse
import json
import os
import platform
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .shared import digest, hash_file, read, signature

ROOT = Path(__file__).resolve().parents[1]
NAME = "prepilot.zip"


def sources():
    fixed = ["README.md","PROTOCOL.md","DATA.md","FAILURE-REVIEW.md","requirements.txt",".gitignore",
             ".github/workflows/test.yml"]
    paths = [ROOT/name for name in fixed]
    for folder in ("rus","scripts","tests"):
        paths.extend(sorted((ROOT/folder).glob("*.py")))
    if any(path.is_symlink() or not path.resolve().is_relative_to(ROOT) for path in paths):
        raise ValueError("Refusing to archive a linked source file")
    return {path.relative_to(ROOT).as_posix():path for path in paths}


def manifest_record(run):
    path = run/"manifest.json"
    manifest = read(path)
    portable = dict(manifest)
    portable["rows"] = [{key:value for key,value in row.items() if key not in ("image","masks")}
                        for row in manifest["rows"]]
    return dict(raw=hash_file(path), portable=signature(portable))


def environment():
    packages = {}
    for line in (ROOT/"requirements.txt").read_text().splitlines():
        package, pinned = line.split("==")
        installed = version(package)
        if installed.split("+",1)[0] != pinned:
            raise ValueError(f"{package}: expected {pinned}, found {installed}")
        packages[package] = installed
    return dict(python=platform.python_version(), os=platform.system(), machine=platform.machine(), packages=packages)


def create(run):
    if (run/"pilot").exists() or (run/"freeze.json").exists():
        raise ValueError("Cannot create a pre-pilot snapshot after pilot artifacts exist")
    if (run/NAME).exists():
        raise FileExistsError("Existing pre-pilot snapshot preserved")
    files = sources()
    meta = dict(schema=1, recorded_utc=datetime.now(timezone.utc).isoformat(),
                environment=environment(), manifest=manifest_record(run),
                files={name:hash_file(path) for name,path in files.items()})
    with tempfile.NamedTemporaryFile(dir=run) as staging:
        with zipfile.ZipFile(staging,"w",compression=zipfile.ZIP_DEFLATED) as archive:
            for name,path in files.items():
                archive.writestr(name,path.read_bytes())
            archive.writestr("record.json",json.dumps(meta,sort_keys=True,indent=2))
        staging.flush()
        os.link(staging.name,run/NAME)
    return verify(run)


def verify(run):
    with zipfile.ZipFile(run/NAME) as archive:
        meta = json.loads(archive.read("record.json"))
        files = sources()
        if (meta["schema"] != 1 or set(meta["files"]) != files.keys()
            or sorted(archive.namelist()) != sorted([*files, "record.json"])):
            raise ValueError("Pre-pilot source inventory changed")
        for name,path in files.items():
            if not hash_file(path) == meta["files"][name] == digest(archive.read(name)):
                raise ValueError(f"Pre-pilot source mismatch: {name}")
    manifest = manifest_record(run)
    if manifest["portable"] != meta["manifest"]["portable"]:
        raise ValueError("Pre-pilot dataset manifest mismatch")
    return dict(archive_sha256=hash_file(run/NAME), manifest_raw_matches=manifest["raw"]==meta["manifest"]["raw"],
                manifest_raw_sha256=manifest["raw"], runtime=environment())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create","verify"))
    parser.add_argument("--run",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps((create if args.action=="create" else verify)(args.run),indent=2))

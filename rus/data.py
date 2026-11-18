import csv
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from openpyxl import load_workbook
from sklearn.model_selection import StratifiedGroupKFold

from .shared import digest, hash_file, save

UCLM_MAP = dict(columns=dict(image="Image", label="Label", doppler="Doppler", marked="Marks", combined="Combined"),
                labels=dict(Benign="benign", Malignant="malignant", Normal="normal"))


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        return list(csv.DictReader(stream, dialect=csv.Sniffer().sniff(sample, delimiters=",;\t")))


def row(cohort, patient, image, masks, label, fold=0):
    image = Path(image).resolve()
    if not str(patient).strip() or patient is None:
        raise ValueError(f"Missing patient key for {image.name}")
    return dict(cohort=cohort, patient=f"{cohort}:{patient}", id=f"{cohort}:{image.stem}",
                image=str(image), masks=[str(Path(mask).resolve()) for mask in masks],
                label=label.lower().strip(), fold=int(fold))


def busbra(root):
    root = Path(root)
    folds = {item["ID"]: item["kFold"] for item in table(root / "5-fold-cv.csv")}
    return [row("busbra", item["Case"], root / "Images" / (item["ID"] + ".png"),
                [root / "Masks" / (item["ID"].replace("bus_", "mask_", 1) + ".png")],
                item["Pathology"], folds.get(item["ID"], 0)) for item in table(root / "bus_data.csv")]


def png_index(root):
    files = {}
    for path in Path(root).rglob("*.png"):
        if path.name.startswith("._"):
            continue
        if path.name in files:
            raise ValueError(f"Ambiguous PNG basename: {path.name}")
        files[path.name] = path
    return files


def breast(root, sheet):
    files = png_index(root)
    book = load_workbook(sheet, read_only=True, data_only=True)
    try:
        stream = iter(book.active.values)
        header = [str(value).strip() for value in next(stream)]
        rows = []
        for values in stream:
            item = dict(zip(header, values))
            if not item["Image_filename"]:
                continue
            masks = str(item["Mask_tumor_filename"] or "").split("&")
            case = item.get("CaseID", item.get("Case_ID"))
            rows.append(row("breast", case, files[item["Image_filename"].strip()],
                            [files[name.strip()] for name in masks if name.strip()],
                            str(item["Classification"])))
        return rows
    finally:
        book.close()


def flag(value):
    token = str(value).strip().lower()
    if token in {"true", "yes", "y", "1", "si", "sí"}:
        return True
    if token in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"Unrecognized boolean {value!r}; normalize metadata explicitly")


def uclm(images, masks, metadata, mapping=None):
    mapping = UCLM_MAP if mapping is None else mapping
    files, targets = png_index(images), png_index(masks)
    accepted, excluded = [], []
    for raw in table(metadata):
        item = {key: raw[column].strip() for key, column in mapping["columns"].items()}
        filename = item["image"]
        label = mapping["labels"][item["label"]]
        reasons = [key for key in ("doppler", "marked", "combined") if flag(item[key])]
        if reasons or (label != "normal" and filename not in targets):
            excluded.append(dict(image=filename, reasons=reasons or ["missing_annotation"]))
            continue
        if filename in targets:
            with Image.open(files[filename]) as image, Image.open(targets[filename]) as target:
                if image.size != target.size:
                    excluded.append(dict(image=filename, reasons=["mask_geometry_mismatch"],
                                         image_size=image.size, mask_size=target.size))
                    continue
        patient = re.fullmatch(r"([A-Za-z]{4})_\d+\.png", filename, re.I)
        if not patient:
            raise ValueError(f"No verified patient prefix: {filename}")
        accepted.append(row("uclm", patient[1], files[filename],
                            [targets[filename]] if filename in targets else [], label))
    return accepted, excluded


def pixels(path, mode):
    with Image.open(path) as source:
        return np.array(source.convert(mode))


def reference(item, shape):
    if item["label"] not in {"benign", "malignant", "normal"}:
        raise ValueError(f"Unknown reference label: {item['label']}")
    target = np.zeros(shape, dtype=bool)
    for path in item["masks"]:
        if item["cohort"] == "uclm":
            rgb = pixels(path, "RGB")
            red = np.all(rgb == [255, 0, 0], axis=2)
            green = np.all(rgb == [0, 255, 0], axis=2)
            if not np.all(red | green | np.all(rgb == 0, axis=2)):
                raise ValueError(f"Unknown RGB mask color: {path}")
            binary = red | green
        else:
            binary = pixels(path, "L") > 0
        if binary.shape != shape:
            raise ValueError(f"Mask geometry mismatch: {path}")
        target |= binary
    if bool(target.any()) != (item["label"] != "normal"):
        raise ValueError(f"Missing or contradictory annotation: {item['id']}")
    return target


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left, right):
        self.parent[self.find(left)] = self.find(right)


def prepare(rows, run, exclusions=(), decisions=None, demo=False):
    if (run / "manifest.json").exists():
        raise ValueError("Use a fresh run directory; manifests are immutable")
    decisions = decisions or {}
    ids, exact, fingerprints = {}, defaultdict(list), {}
    groups = Groups()
    for item in rows:
        key = item["id"]
        if key in ids:
            raise ValueError(f"Duplicate image ID: {key}")
        ids[key] = item
        image = pixels(item["image"], "L")
        reference(item, image.shape)
        item["image_sha"] = hash_file(item["image"])
        item["mask_sha"] = [hash_file(path) for path in item["masks"]]
        exact[digest(str(image.shape).encode() + image.tobytes())].append(key)
        thumb = np.asarray(Image.fromarray(image).resize((9, 8), Image.Resampling.LANCZOS))
        fingerprints[key] = int.from_bytes(np.packbits(thumb[:, 1:] > thumb[:, :-1]).tobytes())
    quarantine, candidates, seen, confirmed, exact_pairs = set(), [], set(), [], set()
    def resolve(left, right):
        first, second = ids[left], ids[right]
        if first["cohort"] == second["cohort"] == "busbra":
            groups.join(first["patient"], second["patient"])
        elif first["cohort"] != second["cohort"]:
            quarantine.update(key for key in (left, right) if ids[key]["cohort"] != "busbra")
        elif first["patient"] != second["patient"]:
            quarantine.update((left, right))
    for keys in exact.values():
        for left, right in combinations(keys, 2):
            resolve(left, right)
            exact_pairs.add(tuple(sorted((left, right))))
    for left, right in combinations(sorted(ids), 2):
        if ids[left]["patient"] == ids[right]["patient"] or (left, right) in exact_pairs:
            continue
        if (fingerprints[left] ^ fingerprints[right]).bit_count() <= 3:
            pair = left + "|" + right
            seen.add(pair)
            if decisions.get(pair) == "duplicate":
                resolve(left, right)
                confirmed.append(pair)
            elif decisions.get(pair) != "distinct":
                candidates.append(pair)
    if set(decisions) - seen:
        raise ValueError("Duplicate review contains unknown candidate pairs")
    save(run / "duplicate-review.json", dict(pending=candidates, confirmed=confirmed,
         exact=[keys for keys in exact.values() if len(keys) > 1], quarantine=sorted(quarantine)))
    if candidates:
        raise ValueError("Review pending pairs in duplicate-review.json; rerun prepare with --decisions")
    rows = [item for item in rows if item["id"] not in quarantine]
    source = [item for item in rows if item["cohort"] == "busbra"]
    assignments = defaultdict(set)
    for item in source:
        assignments[groups.find(item["patient"])].add(item["fold"])
    fallback = any(len(folds) != 1 for folds in assignments.values()) or {r["fold"] for r in source} != set(range(1, 6))
    if fallback:
        split = StratifiedGroupKFold(5, shuffle=True, random_state=2411)
        for fold, (_, indices) in enumerate(split.split(source, [r["label"] for r in source],
                                            [groups.find(r["patient"]) for r in source]), 1):
            for index in indices:
                source[index]["fold"] = fold
    if {r["fold"] for r in source} != set(range(1, 6)):
        raise ValueError("Five nonempty source folds required")
    save(run / "manifest.json", dict(demo=demo, rows=rows, split="regenerated" if fallback else "published",
                                     excluded=list(exclusions), quarantine=sorted(quarantine)))
    counts = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        members = [r for r in rows if r["cohort"] == cohort]
        counts[cohort] = dict(scans=len(members), patients=len({r["patient"] for r in members}),
                              normals=sum(r["label"]=="normal" for r in members),
                              folds={fold:sum(r["fold"]==fold for r in members) for fold in sorted({r["fold"] for r in members})})
    save(run / "counts.json", counts)

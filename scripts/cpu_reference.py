"""Measure the development-only Dice attainable from lesion position alone."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from rus.shared import hash_file, read, save


def development(rows):
    owners = {}
    selected = []
    for row in rows:
        if row["cohort"] != "busbra":
            continue
        patient, fold = row["patient"], row["fold"]
        if owners.setdefault(patient, fold) != fold:
            raise ValueError("BUS-BRA patient crosses folds")
        if fold in (1, 2, 3):
            selected.append(row)
    if {row["fold"] for row in selected} != {1, 2, 3}:
        raise ValueError("All three development folds are required")
    return selected


def native_mask(row):
    if len(row["masks"]) != 1 or row["label"] not in {"benign", "malignant"}:
        raise ValueError("Expected one lesion annotation for each BUS-BRA scan")
    path, = row["masks"]
    if [hash_file(path)] != row["mask_sha"]:
        raise ValueError("Annotation differs from the prepared manifest")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if not mask.any():
        raise ValueError("Empty BUS-BRA lesion annotation")
    return mask


def resize(mask, size):
    image = Image.fromarray(mask.astype(np.uint8))
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=bool)


def evaluate(rows):
    selected = development(rows)
    grids, counts = {}, defaultdict(int)
    for row in selected:
        patient = row["patient"]
        if patient not in grids:
            grids[patient] = np.zeros((128, 128), dtype=np.float64)
        grids[patient] += resize(native_mask(row), (128, 128))
        counts[patient] += 1
    folds, all_scores = [], []
    for fold in (1, 2, 3):
        training = sorted({row["patient"] for row in selected if row["fold"] != fold})
        template = np.mean([grids[patient]/counts[patient] for patient in training], axis=0) >= .5
        scores = defaultdict(list)
        misses = 0
        for row in selected:
            if row["fold"] != fold:
                continue
            target = native_mask(row)
            prediction = resize(template, target.shape[::-1])
            intersection = np.count_nonzero(prediction & target)
            misses += int(intersection == 0)
            scores[row["patient"]].append(2*intersection/int(prediction.sum()+target.sum()))
        patient_means = [float(np.mean(values)) for values in scores.values()]
        all_scores.extend(patient_means)
        folds.append(dict(held_out_fold=fold, training_patients=len(training), patients=len(scores),
                          scans=sum(map(len, scores.values())), missed_scans=misses,
                          template_area_fraction=float(template.mean()), patient_macro_dice=float(np.mean(patient_means))))
    return dict(method="Patient-balanced positional prior; 128x128 normalized grid; fixed threshold 0.5",
                scope="BUS-BRA development only; not a routing result or pilot decision",
                scans=len(selected), patients=len(all_scores), patient_macro_dice=float(np.mean(all_scores)), folds=folds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.run/"manifest.json")
    if manifest["demo"]:
        parser.error("Use a prepared clinical manifest for this development reference")
    result = evaluate(manifest["rows"])
    path = args.run/"cpu-reference.json"
    save(path, result)
    print(f'{path}: {result["scans"]} scans, {result["patients"]} patients, Dice {result["patient_macro_dice"]:.4f}')


if __name__ == "__main__":
    main()

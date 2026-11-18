"""Build a private, development-only qualitative worksheet from completed OOF predictions."""
import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from rus.data import pixels, reference
from rus.shared import hash_file, patients, read, stable
from scripts.cpu_reference import development


CATEGORIES = ("rescued", "lost_overlap", "both_missed", "lower_dice", "higher_dice", "unchanged")


def category(record):
    if not record["positive"]:
        raise ValueError("Expected lesion-positive BUS-BRA development scans")
    if record["first_miss"]:
        return "both_missed" if record["second_miss"] else "rescued"
    if record["second_miss"]:
        return "lost_overlap"
    if record["second"] == record["first"]:
        return "unchanged"
    return "higher_dice" if record["second"] > record["first"] else "lower_dice"


def cases(run, rows):
    selected = development(rows)
    records = []
    for fold in (1, 2, 3):
        folder = run/"crossfit"/str(fold)
        if not (folder/"provenance.json").is_file():
            raise ValueError(f"Development fold {fold} is not ready; no other predictions may be substituted")
        expected = {item["id"]: item for item in selected if item["fold"] == fold}
        provenance = read(folder/"provenance.json")
        if (provenance["folds"] != sorted({1, 2, 3}-{fold})
            or provenance["training_patients"] != patients([item for item in selected if item["fold"] != fold])
            or provenance["heldout_patients"] != patients(expected.values())
            or provenance["predictions"] != hash_file(folder/"predictions.json")):
            raise ValueError(f"Invalid out-of-fold provenance: fold {fold}")
        predictions = read(folder/"predictions.json")
        if len(predictions) != len(expected) or {row["id"] for row in predictions} != expected.keys():
            raise ValueError(f"Incomplete or unexpected scan IDs: fold {fold}")
        for record in predictions:
            source = expected[record["id"]]
            if record["patient"] != source["patient"] or record["fold"] != fold:
                raise ValueError(f"Patient or fold mismatch: fold {fold}")
            records.append((category(record), record, source))
    return records


def sample(records, limit):
    chosen = []
    for group in CATEGORIES:
        seen = set()
        for item in sorted((item for item in records if item[0] == group),
                           key=lambda item: (stable("review/"+item[1]["id"]), item[1]["id"])):
            patient = item[1]["patient"]
            if patient not in seen and len(seen) < limit:
                chosen.append(item)
                seen.add(patient)
    return chosen


def panel(run, record, source, output):
    if hash_file(source["image"]) != source["image_sha"] or [hash_file(path) for path in source["masks"]] != source["mask_sha"]:
        raise ValueError("Source data changed after manifest creation")
    frame = pixels(source["image"], "L")
    target = reference(source, frame.shape)
    edge = target ^ binary_erosion(target)
    path = run/"crossfit"/str(source["fold"])/"masks"/f'busbra-{stable(source["id"])}.npz'
    panes = []
    with np.load(path, allow_pickle=False) as masks:
        for name in ("reference", "first", "second"):
            rgb = np.repeat(frame[:, :, None], 3, axis=2)
            rgb[edge] = [0, 255, 80]
            if name != "reference":
                mask = masks[name]
                if mask.dtype != bool or mask.shape != target.shape:
                    raise ValueError("Invalid native mask geometry or type")
                intersection = np.count_nonzero(mask & target)
                dice = 2*intersection/int(mask.sum()+target.sum())
                if (not np.isclose(dice, record[name], atol=1e-10, rtol=0)
                    or bool(mask.any()) != record[name+"_nonempty"]
                    or (intersection == 0) != record[name+"_miss"]):
                    raise ValueError("Saved mask does not match recorded outcome")
                outline = mask ^ binary_erosion(mask)
                rgb[outline] = [255, 0, 210]
                rgb[edge & outline] = 255
            image = Image.fromarray(rgb)
            image.thumbnail((480, 480))
            panes.append(image)
    width, height = panes[0].size
    sheet = Image.new("RGB", (3*width, height+24))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(panes, ("Reference", "First", "Second"))):
        sheet.paste(image, (index*width, 24))
        draw.text((index*width+2, 3), label, fill="white")
    sheet.save(output)


def generate(run, limit=4):
    if limit < 1:
        raise ValueError("Per-category sample must be positive")
    output = run/"failure-review"
    if output.exists():
        raise FileExistsError("Existing worksheet preserved; move it before generating a new copy")
    manifest = read(run/"manifest.json")
    records = cases(run, manifest["rows"])
    chosen = sample(records, limit)
    counts = Counter(item[0] for item in records)
    title = "Synthetic rehearsal — not study evidence" if manifest["demo"] else "BUS-BRA development review"
    lines = [f"# {title}", "", "Private worksheet. Development folds 1–3 only. Reviewer: ____  Date: ____",
             "", "## All eligible scans (not just the selected panels)", "", "| Outcome | Scans |", "|---|---:|"]
    lines += [f"| {group} | {counts[group]} |" for group in CATEGORIES]
    lines += ["", f"{len(chosen)} sampled cards from {len(records)} scans. Samples are not prevalence estimates.",
              "Reference = green; prediction = magenta; coincident boundaries = white."]
    with tempfile.TemporaryDirectory(dir=run) as staging:
        packet = Path(staging)/"packet"
        packet.mkdir()
        for index, (group, record, source) in enumerate(chosen, 1):
            alias = f"case-{index:03}"
            panel(run, record, source, packet/(alias+".png"))
            lines += ["", f"## {alias}: {group}", "", f"![Reference and predictions]({alias}.png)", "",
                      "| Pass | Dice | Empty? | Miss? |", "|---|---:|---|---|"]
            for name in ("first", "second"):
                lines.append(f'| {name} | {record[name]:.4f} | {not record[name+"_nonempty"]} | {record[name+"_miss"]} |')
            lines += ["", "- Visible change (undercoverage / overcoverage / displacement / disconnected component / none / unclear): ____",
                      "- Location and evidence: ____", "- Confidence (clear / uncertain / not assessable): ____"]
        (packet/"WORKSHEET.md").write_text("\n".join(lines)+"\n")
        packet.rename(output)
    return output/"WORKSHEET.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    args = parser.parse_args()
    print(generate(args.run, args.per_category))

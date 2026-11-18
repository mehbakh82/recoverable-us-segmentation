"""Post-hoc CPU audit. Reads only the entropy-selected fold-4 mask pairs; emits aggregates."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from rus import snapshot
from rus.data import reference
from rus.routing import gate, selected
from rus.shared import hash_file, read, stable
from rus.vision import crop_box, cut


EDITS = ("lesion_recovered", "lesion_erased", "false_positive_removed", "false_positive_added")


def measure(first, second, target):
    if any(mask.shape != target.shape or mask.dtype != bool for mask in (first, second, target)):
        raise ValueError("Invalid native mask geometry or dtype")
    area = int(target.sum())
    if not area:
        raise ValueError("Expected a positive reference mask")
    box = crop_box(first)
    if int(cut(second, box).sum()) != int(second.sum()):
        raise ValueError("Refiner foreground lies outside the reconstructed crop")
    added, removed = second & ~first, first & ~second
    edits = [int((added & target).sum()), int((removed & target).sum()),
             int((removed & ~target).sum()), int((added & ~target).sum())]
    first_dice = 2*int((first & target).sum())/(int(first.sum())+area)
    denom = int(second.sum())+area
    second_dice = 2*int((second & target).sum())/denom
    lesion_term = (2-first_dice)*(edits[0]-edits[1])/denom
    background_term = first_dice*(edits[2]-edits[3])/denom
    if not math.isclose(second_dice-first_dice, lesion_term+background_term, abs_tol=1e-12):
        raise ValueError("Pixel changes do not reconstruct the native Dice difference")
    inside = int(cut(target,box).sum())
    ceiling = 2*inside/(area+inside)
    if ceiling+1e-12 < max(first_dice,second_dice):
        raise ValueError("Infeasible crop ceiling")
    return dict(**dict(zip(EDITS, [value/area for value in edits])), first=first_dice, second=second_dice,
                lesion_term=lesion_term, background_term=background_term,
                outside_fraction=(area-inside)/area, ceiling=ceiling, headroom=ceiling-first_dice)


def aggregate(group):
    if not group:
        return {"scans": 0}
    fields = (*EDITS, "outside_fraction", "ceiling", "headroom", "lesion_term", "background_term")
    return dict(scans=len(group), with_lesion_outside=sum(item["outside_fraction"]>0 for item in group),
                net_lesion_loss=sum(item["lesion_term"]<0 for item in group),
                net_fp_increase=sum(item["background_term"]<0 for item in group),
                more_adverse_term=dict(Counter("equal" if math.isclose(item["lesion_term"],item["background_term"],abs_tol=1e-12)
                                               else "lesion" if item["lesion_term"]<item["background_term"] else "background"
                                               for item in group)),
                quantiles={key:np.quantile([item[key] for item in group],[.25,.5,.75]).tolist() for key in fields})


def audit(run):
    recorded = read(run/"pilot/gate.json")
    if recorded["demo"] or recorded["passed"] or snapshot.verify(run) != recorded["prepilot"]:
        raise ValueError("Not the verified stopped clinical pilot")
    path = run/"pilot/predictions.json"
    if hash_file(path) != recorded["predictions"]:
        raise ValueError("Pilot predictions changed")
    rows = read(path)
    sources = {item["id"]:item for item in read(run/"manifest.json")["rows"] if item["cohort"]=="busbra" and item["fold"]==4}
    if (len(rows)!=len(sources) or len(rows)!=len({item["id"] for item in rows})
        or any(item["id"] not in sources or item["patient"]!=sources[item["id"]]["patient"] or item["fold"]!=4
               or not item["positive"] for item in rows)
        or any(recorded[key]!=value for key,value in gate(rows).items())):
        raise ValueError("Fold membership or gate replay mismatch")
    chosen = selected(rows,"entropy",.2)
    changes = {"improved":[], "worsened":[], "unchanged":[]}
    for row in rows:
        if row["id"] not in chosen:
            continue
        source = sources[row["id"]]
        if [hash_file(mask) for mask in source["masks"]] != source["mask_sha"]:
            raise ValueError("Reference mask checksum mismatch")
        with np.load(run/"pilot/masks"/f'busbra-{stable(row["id"])}.npz',allow_pickle=False) as masks:
            first, second = masks["first"], masks["second"]
            result = measure(first, second, reference(source,first.shape))
        if row["first"] != result["first"] or row["second"] != result["second"]:
            raise ValueError("Saved masks do not reproduce both recorded Dice values")
        changes["improved" if result["second"]>result["first"] else
                "worsened" if result["second"]<result["first"] else "unchanged"].append(result)
    return dict(scope="Post hoc, BUS-BRA fold 4, entropy-selected retries only", n=len(chosen),
                patients=len({row["patient"] for row in rows if row["id"] in chosen}),
                units="Native pixel edits / reference lesion area per scan; quantiles are [Q1, median, Q3]",
                unchanged=len(changes["unchanged"]), improved=aggregate(changes["improved"]),
                worsened=aggregate(changes["worsened"]), all_selected=aggregate(sum(changes.values(), [])),
                predictions_sha256=recorded["predictions"], archive_sha256=recorded["prepilot"]["archive_sha256"])


if __name__=="__main__":
    print(json.dumps(audit(Path(sys.argv[1])),sort_keys=True,indent=2))

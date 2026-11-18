"""Aggregate-only CPU closeout. Run from the frozen project root with PYTHONPATH=."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

from rus import snapshot
from rus.routing import gate, selected
from rus.shared import hash_file, patients, read


AGGREGATES = ("pilot/gate.json", "report/REPORT.md", "hardware.json",
              "models/development/first.json", "models/development/refine.json",
              "models/development/hardware.json")


def outcomes(rows, chosen, counts):
    subset = [row for row in rows if row["id"] in chosen]
    changes = [row["second"]-row["first"] for row in subset]
    return dict(scans=len(subset), helped=sum(change>0 for change in changes),
                harmed=sum(change<0 for change in changes), unchanged=changes.count(0.),
                rescued=sum(row["first_miss"] and not row["second_miss"] for row in subset),
                lost_overlap=sum(not row["first_miss"] and row["second_miss"] for row in subset),
                macro_dice_contribution=sum(change/counts[row["patient"]]
                                            for change,row in zip(changes,subset))/len(counts))


def summary(run):
    pilot = run/"pilot"
    manifest = read(run/"manifest.json")
    if manifest["demo"]:
        raise ValueError("Expected clinical pilot, not synthetic")
    receipt = snapshot.verify(run)
    recorded = read(pilot/"gate.json")
    provenance = read(pilot/"provenance.json")
    if receipt != recorded["prepilot"] or recorded["demo"]:
        raise ValueError("Snapshot or scoring runtime differs from the pilot receipt")
    path = pilot/"predictions.json"
    if not hash_file(path) == recorded["predictions"] == provenance["predictions"]:
        raise ValueError("Predictions digest mismatch")
    source = [row for row in manifest["rows"] if row["cohort"]=="busbra"]
    training = [row for row in source if row["fold"] in (1,2,3)]
    heldout = [row for row in source if row["fold"]==4]
    rows = read(path)
    if (provenance["folds"] != [1,2,3] or provenance["training_patients"] != patients(training)
        or provenance["heldout_patients"] != patients(heldout) or set(patients(training)) & set(patients(heldout))
        or sorted((r["id"],r["patient"],r["fold"]) for r in rows) !=
           sorted((r["id"],r["patient"],r["fold"]) for r in heldout)
        or len({r["id"] for r in rows}) != len(rows) or not all(r["positive"] for r in rows)):
        raise ValueError("Not the complete, patient-disjoint, lesion-positive fold-4 panel")
    for kind in ("first","refine"):
        base = run/"models"/"development"
        if hash_file(base/f"{kind}.pt") != provenance["models"][kind]:
            raise ValueError("Checkpoint does not match prediction provenance")
        if [r["epoch"] for r in read(base/f"{kind}.json")] != list(range(1,51)):
            raise ValueError("Incomplete epoch ledger")
    replay = gate(rows)
    if any(recorded[key] != value for key,value in replay.items()) or replay["passed"]:
        raise ValueError("Expected exact replay of the recorded stop decision")
    entropy = selected(rows,"entropy",.2)
    oracle = selected(rows,"oracle",.2)
    counts = Counter(row["patient"] for row in rows)
    if len(entropy) != len(oracle) or len(oracle) != len(rows)//5:
        raise ValueError("Retry budgets differ")
    groups = {name:outcomes(rows, subset, counts) for name,subset in
              (("entropy",entropy),("oracle",oracle),("shared",entropy & oracle),
               ("entropy_only",entropy-oracle),("oracle_only",oracle-entropy))}
    unique_delta = groups["oracle_only"]["macro_dice_contribution"]-groups["entropy_only"]["macro_dice_contribution"]
    if not math.isclose(unique_delta,replay["oracle_minus_entropy"],abs_tol=1e-12):
        raise ValueError("Exclusive retries do not explain the score difference")
    return dict(kind="descriptive fold-4 closeout; no model selection", scans=len(rows), patients=len(counts),
                groups=groups, gate=replay, archive_sha256=receipt["archive_sha256"],
                predictions_sha256=recorded["predictions"], portable_manifest_verified=True,
                aggregate_sha256={name:hash_file(run/name) for name in AGGREGATES})


if __name__ == "__main__":
    print(json.dumps(summary(Path(sys.argv[1])),sort_keys=True,indent=2))

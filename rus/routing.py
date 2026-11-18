import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .shared import mean, stable
from .vision import FEATURES

BUDGETS = (0., 0.05, 0.1, 0.2, 0.4, 1.)
POLICIES = ("never", "always", "random", "entropy", "gain", "difficulty", "oracle", "threshold")


def fit(rows, objective="gain"):
    matrix = [[r["features"][name] for name in FEATURES] for r in rows]
    if objective == "gain":
        target = [r["second"] - r["first"] for r in rows]
    elif objective == "difficulty":
        target = [1. - r["first"] for r in rows]
    else:
        raise ValueError(f"Unknown routing objective: {objective}")
    estimator = GradientBoostingRegressor(n_estimators=60, max_depth=2, min_samples_leaf=5,
                                          learning_rate=0.05, random_state=2411).fit(matrix, target)
    trees = []
    for stage in estimator.estimators_:
        tree = stage[0].tree_
        trees.append(dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                          feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                          value=tree.value[:, 0, 0].tolist()))
    return dict(objective=objective, features=list(FEATURES), start=float(estimator.init_.constant_.item()),
                rate=estimator.learning_rate, trees=trees)


def score(model, features):
    values = [np.float32(features[name]) for name in model["features"]]
    result = model["start"]
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            branch = "left" if float(values[tree["feature"][node]]) <= tree["threshold"][node] else "right"
            node = tree[branch][node]
        result += model["rate"] * tree["value"][node]
    return result


def order(row, policy, counts):
    if policy == "oracle":
        return (row["second"]-row["first"])/counts[row["patient"]] if row["positive"] else 0.
    if policy == "entropy":
        return row["features"]["entropy"]
    if policy == "random":
        return stable("2411/random/"+row["id"])
    return row[policy]


def ranked(rows, policy):
    counts = Counter(row["patient"] for row in rows if row["positive"]) if policy == "oracle" else {}
    return sorted(rows, key=lambda row: (-order(row, policy, counts), stable(row["id"]), row["id"]))


def calibrate(rows):
    ordered = ranked(rows, "gain")
    count = len(rows)//5
    if count == 0:
        raise ValueError("Calibration requires at least five scans")
    last = ordered[count-1]
    return dict(value=last["gain"], tie=stable(last["id"]))


def above(value, key, threshold):
    return value > threshold["value"] or (value == threshold["value"] and stable(key) <= threshold["tie"])


def selected(rows, policy, budget, threshold=None):
    if policy == "never":
        return set()
    if policy == "always":
        return {row["id"] for row in rows}
    if policy == "threshold":
        return {row["id"] for row in rows if above(row["gain"], row["id"], threshold)}
    return {row["id"] for row in ranked(rows, policy)[:math.floor(budget*len(rows))]}


def patient_scores(rows, chosen):
    values = defaultdict(list)
    for row in rows:
        if row["positive"]:
            values[row["patient"]].append(row["second"] if row["id"] in chosen else row["first"])
    return {key:float(np.mean(scores)) for key, scores in values.items()}


def summarize(rows, chosen, policy):
    positives = [row for row in rows if row["positive"]]
    normals = [row for row in rows if not row["positive"]]
    retried = [row for row in positives if row["id"] in chosen]
    def field(row, name):
        return row[("second" if row["id"] in chosen else "first")+name]
    return dict(scans=len(rows), patients=len({r["patient"] for r in rows}), positives=len(positives),
                normals=len(normals), retries=len(chosen), retry_rate=len(chosen)/len(rows),
                patient_dice=mean(list(patient_scores(rows, chosen).values())),
                helped_retried_positive=mean([row["second"]>row["first"] for row in retried]),
                harmed_retried_positive=mean([row["second"]<row["first"] for row in retried]),
                missed_positive=sum(field(row, "_miss") for row in positives),
                normal_fp_rate=mean([field(row, "_nonempty") for row in normals]),
                replay_mean_ms=mean([row["first_ms"]+(row["router_ms"] if policy in ("gain","threshold") else
                                row["difficulty_ms"] if policy=="difficulty" else 0)
                                +(row["second_ms"] if row["id"] in chosen else 0) for row in rows]))


def gate(rows):
    results = {name:mean(list(patient_scores(rows, selected(rows, name, .2)).values()))
               for name in ("never","entropy","oracle")}
    if None in results.values():
        raise ValueError("No positive patients for pilot")
    gap = results["oracle"]-results["entropy"]
    improvement = results["oracle"]-results["never"]
    tolerance = 8 * math.ulp(1.)
    gains = [r["second"]-r["first"] for r in rows if r["positive"]]
    return dict(scores=results, oracle_minus_entropy=gap, oracle_minus_never=improvement,
                improved_fraction=mean([value>0 for value in gains]), harmed_fraction=mean([value<0 for value in gains]),
                passed=gap>=0.01-tolerance and improvement>tolerance)


def contrast(rows, chosen, comparator, reps):
    gain = patient_scores(rows, chosen)
    baseline = patient_scores(rows, comparator)
    diff = np.array([gain[key]-baseline[key] for key in sorted(gain)])
    if not len(diff):
        raise ValueError("No positive patients for primary endpoint")
    rng = np.random.default_rng(2411)
    samples = [rng.choice(diff, size=len(diff)).mean() for _ in range(reps)]
    return dict(delta=float(diff.mean()), interval=np.quantile(samples,[.025,.975]).tolist(),
                positive_patients=len(diff))


def analyze(rows, threshold, reps):
    choices = {policy:selected(rows, policy, .2, threshold) for policy in POLICIES}
    main = {policy:summarize(rows, chosen, policy) for policy, chosen in choices.items()}
    primary = contrast(rows, choices["gain"], choices["entropy"], reps)
    secondary = contrast(rows, choices["gain"], choices["difficulty"], reps)
    secondary.update(common=len(choices["gain"] & choices["difficulty"]),
                     gain_only=len(choices["gain"] - choices["difficulty"]),
                     difficulty_only=len(choices["difficulty"] - choices["gain"]))
    curves = {policy:[dict(budget=budget, **summarize(rows, selected(rows, policy, budget), policy))
                      for budget in BUDGETS] for policy in ("gain","difficulty","entropy","random","oracle")}
    return dict(policies=main, **primary, gain_vs_difficulty=secondary, curves=curves)

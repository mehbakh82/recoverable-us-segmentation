import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .data import pixels, reference
from .routing import above, calibrate, fit, gate, score
from .shared import config, hash_file, patients, read, save, signature, stable
from . import snapshot
from .vision import first_pass, load_model, second_pass, select_device, smoke, sync, train_one


def code_files():
    return sorted(Path(__file__).parent.glob("*.py"))


class Study:
    def __init__(self, run):
        self.run = Path(run)
        manifest = read(self.run / "manifest.json")
        self.rows = manifest["rows"]
        self.cfg = config(manifest["demo"])
        self.demo = manifest["demo"]
        self.source = [row for row in self.rows if row["cohort"] == "busbra"]
        self.development = [row for row in self.source if row["fold"] <= 3]
        self.calibration = [row for row in self.source if row["fold"] == 4]
        self.internal = [row for row in self.source if row["fold"] == 5]
        sets = [set(patients([r for r in self.source if r["fold"] == fold])) for fold in range(1,6)]
        if any(not group or any(group & other for other in sets[:index]) for index,group in enumerate(sets)):
            raise ValueError("Source fold patient isolation failed")
        self.pair = self.run / "models" / "development"

    def mutable(self):
        if (self.run / "freeze.json").exists():
            raise ValueError("Study frozen: a new protocol requires a new run")

    def verify_inputs(self, rows):
        for row in rows:
            if hash_file(row["image"]) != row["image_sha"] or [hash_file(path) for path in row["masks"]] != row["mask_sha"]:
                raise ValueError(f"Input changed: {row['id']}")

    def profile(self):
        self.mutable()
        device = select_device(self.demo)
        smoke(self.cfg, device, self.run / "hardware.json")

    def training(self, rows, folder):
        device = select_device(self.demo)
        micro = smoke(self.cfg, device, folder / "hardware.json")
        for kind in ("first","refine"):
            while True:
                try:
                    train_one(rows, kind, self.cfg, device, micro, folder/f"{kind}.pt")
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise RuntimeError("Out of memory at microbatch 1; paused at last completed epoch") from None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                micro //= 2

    def train(self):
        self.mutable()
        self.verify_inputs(self.development)
        self.training(self.development, self.pair)

    def models(self, folder, device, training, heldout):
        if set(patients(training)) & set(patients(heldout)):
            raise ValueError("Training and inference patients overlap")
        result = []
        for kind in ("first","refine"):
            model, meta = load_model(folder/f"{kind}.pt", self.cfg, device)
            if meta["patients"] != patients(training) or meta["kind"] != kind or meta["inputs"] != signature(
                [(r["id"],r["image_sha"],r["mask_sha"]) for r in training]):
                raise ValueError("Prediction provenance does not match training inputs")
            result.append(model)
        return result

    def collect(self, rows, training, folder, destination, settings=None):
        device = select_device(self.demo)
        first, refine = self.models(folder, device, training, rows)
        dataset = []
        blank = np.zeros((100, 160), dtype=np.uint8)
        _, box, _ = first_pass(first, blank, self.cfg, device)
        second_pass(refine, blank, box, self.cfg, device)
        for row in rows:
            image = pixels(row["image"], "L")
            sync(device)
            started = time.perf_counter()
            first_mask, box, inputs = first_pass(first, image, self.cfg, device)
            sync(device)
            first_ms = 1000*(time.perf_counter()-started)
            started = time.perf_counter()
            gain = score(settings["model"], inputs) if settings else None
            router_ms = 1000*(time.perf_counter()-started) if settings else 0.
            started = time.perf_counter()
            difficulty = score(settings["difficulty_model"], inputs) if settings else None
            difficulty_ms = 1000*(time.perf_counter()-started) if settings else 0.
            started = time.perf_counter()
            refined = second_pass(refine, image, box, self.cfg, device)
            sync(device)
            second_ms = 1000*(time.perf_counter()-started)
            target = reference(row, image.shape)
            positive = bool(target.any())
            record = dict(id=row["id"], patient=row["patient"], fold=row["fold"], features=inputs,
                          gain=gain, difficulty=difficulty, positive=positive, first_ms=first_ms,
                          second_ms=second_ms, router_ms=router_ms, difficulty_ms=difficulty_ms)
            for name, mask in (("first", first_mask), ("second", refined)):
                overlap = int((target & mask).sum())
                denom = int(target.sum()+mask.sum())
                record[name] = 2*overlap/denom if denom else 1.
                record[name+"_nonempty"] = bool(mask.any())
                record[name+"_miss"] = positive and not overlap
            dataset.append(record)
            mask_path = destination / "masks" / (row["cohort"]+"-"+str(stable(row["id"]))+".npz")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, first=first_mask, second=refined)
        save(destination/"predictions.json", dataset)
        save(destination/"provenance.json", dict(folds=sorted({r["fold"] for r in training}),
             training_patients=patients(training), heldout_patients=patients(rows), predictions=hash_file(destination/"predictions.json"),
             models={kind:hash_file(folder/f"{kind}.pt") for kind in ("first","refine")}))
        return dataset

    def pilot(self):
        self.mutable()
        receipt = snapshot.verify(self.run)
        self.verify_inputs(self.calibration)
        out = self.run/"pilot"
        predictions = self.collect(self.calibration, self.development, self.pair, out)
        result = gate(predictions)
        save(out/"gate.json", dict(**result, demo=self.demo, prepilot=receipt, predictions=hash_file(out/"predictions.json")))
        return result

    def passed(self):
        path = self.run/"pilot"
        evidence = read(path/"gate.json")
        if evidence["predictions"] != hash_file(path/"predictions.json") or gate(read(path/"predictions.json"))["passed"] != evidence["passed"]:
            raise ValueError("Pilot evidence changed")
        actual = {kind:hash_file(self.pair/f"{kind}.pt") for kind in ("first","refine")}
        if actual != read(path/"provenance.json")["models"]:
            raise ValueError("Pilot checkpoints changed")
        if not self.demo and not evidence["passed"]:
            raise ValueError("Clinical study stopped at pilot; do not open locked cohorts")
        if evidence["prepilot"]["archive_sha256"] != hash_file(self.run/snapshot.NAME):
            raise ValueError("Pre-pilot archive does not match the scored pilot")

    def crossfit(self):
        self.mutable()
        self.passed()
        self.verify_inputs(self.development)
        for fold in (1,2,3):
            training = [r for r in self.development if r["fold"] != fold]
            heldout = [r for r in self.development if r["fold"] == fold]
            folder = self.run/"crossfit"/str(fold)
            self.training(training, folder)
            self.collect(heldout, training, folder, folder)

    def router(self):
        self.mutable()
        self.passed()
        rows = []
        for fold in (1,2,3):
            folder = self.run/"crossfit"/str(fold)
            provenance = read(folder/"provenance.json")
            heldout = read(folder/"predictions.json")
            expected = {r["id"]:r["patient"] for r in self.development if r["fold"]==fold}
            actual = {r["id"]:r["patient"] for r in heldout}
            training = [r for r in self.development if r["fold"]!=fold]
            if (len(actual)!=len(heldout) or actual!=expected or provenance["predictions"]!=hash_file(folder/"predictions.json")
                or provenance["training_patients"] != patients(training)
                or set(actual.values()) & set(provenance["training_patients"])
                or [hash_file(folder/f"{kind}.pt") for kind in ("first","refine")] !=
                   [provenance["models"][kind] for kind in ("first","refine")]):
                raise ValueError("Invalid out-of-fold provenance")
            rows.extend(heldout)
        model = fit(rows)
        calibration = read(self.run/"pilot"/"predictions.json")
        for row in calibration:
            row["gain"] = score(model,row["features"])
        save(self.run/"router.json", dict(model=model, difficulty_model=fit(rows,"difficulty"), threshold=calibrate(calibration)))

    def fingerprint(self):
        paths = [snapshot.NAME,"manifest.json","router.json","pilot/gate.json","duplicate-review.json",
                 "models/development/first.pt","models/development/refine.pt"]
        root = Path(__file__).resolve().parents[1]
        return dict(config=self.cfg, files={path:hash_file(self.run/path) for path in paths},
                    protocol={key:hash_file(root/key) for key in ("PROTOCOL.md","DATA.md","requirements.txt")},
                    source={path.name:hash_file(path) for path in code_files()},
                    primary="breast: patient-macro positive Dice; gain minus entropy; fixed 20% all scans",
                    secondary="breast: patient-macro positive Dice; gain minus difficulty; same 20% all scans",
                    environment=dict(python=platform.python_version(),
                                     packages={name:version(name) for name in ("torch","numpy","pillow","scipy","scikit-learn","openpyxl")}))

    def freeze(self):
        self.mutable()
        self.passed()
        if {r["cohort"] for r in self.rows} != {"busbra","breast","uclm"}:
            raise ValueError("Both external manifests must be prepared before freezing the MSc study")
        if read(self.run/"duplicate-review.json")["pending"]:
            raise ValueError("Duplicate review is incomplete")
        save(self.run/"freeze.json", self.fingerprint())

    def evaluate(self):
        frozen = read(self.run/"freeze.json")
        if frozen != self.fingerprint():
            raise ValueError("Frozen inputs, code, configuration or environment changed")
        settings = read(self.run/"router.json")
        out = self.run/"evaluation"
        if (out/"complete.json").exists():
            raise ValueError("Frozen evaluation already completed; use report")
        cohorts = dict(internal=self.internal, breast=[r for r in self.rows if r["cohort"]=="breast"],
                       uclm=[r for r in self.rows if r["cohort"]=="uclm"])
        self.verify_inputs(self.rows)
        for cohort, rows in cohorts.items():
            predictions = self.collect(rows, self.development, self.pair, out/cohort, settings)
            self.benchmark(rows, out/cohort, settings, predictions)
        save(out/"complete.json", dict(freeze=signature(frozen), predictions={
            cohort:hash_file(out/cohort/"predictions.json") for cohort in cohorts},
            timing={cohort:hash_file(out/cohort/"timing.json") for cohort in cohorts}))

    def benchmark(self, rows, out, settings, predictions):
        device = select_device(self.demo)
        first, refine = self.models(self.pair,device,self.development,rows)
        blank = np.zeros((100,160),dtype=np.uint8)
        _, box, inputs = first_pass(first,blank,self.cfg,device)
        score(settings["model"],inputs)
        second_pass(refine,blank,box,self.cfg,device)
        durations, selected = [], []
        for row, expected in zip(rows, predictions):
            sync(device)
            started = time.perf_counter()
            image = pixels(row["image"],"L")
            _, box, inputs = first_pass(first,image,self.cfg,device)
            retry = above(score(settings["model"],inputs),row["id"],settings["threshold"])
            if retry:
                second_pass(refine,image,box,self.cfg,device)
                selected.append(row["id"])
            sync(device)
            durations.append(1000*(time.perf_counter()-started))
            if retry != above(expected["gain"],row["id"],settings["threshold"]):
                raise ValueError("Deployed and replay decisions differ")
        save(out/"timing.json",dict(policy="source-calibrated threshold", includes_decode=True,
             batch_size=1, retries=len(selected), count=len(rows), mean_ms=float(np.mean(durations)),
             p50_ms=float(np.percentile(durations,50)),p95_ms=float(np.percentile(durations,95))))

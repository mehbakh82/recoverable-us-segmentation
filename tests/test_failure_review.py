import tempfile
import unittest
from pathlib import Path

from rus.shared import hash_file, save
from scripts.failure_review import cases, category, generate, sample


class ReviewTests(unittest.TestCase):
    def test_exclusive_outcomes(self):
        scenarios = [(0., .1, "rescued"), (.1, 0., "lost_overlap"), (0., 0., "both_missed"),
                     (.3, .2, "lower_dice"), (.2, .3, "higher_dice"), (.3, .3, "unchanged")]
        for first, second, expected in scenarios:
            record = dict(positive=True, first=first, second=second, first_miss=first == 0, second_miss=second == 0)
            self.assertEqual(category(record), expected)

    def test_sample_is_deterministic_and_patient_balanced(self):
        records = [("rescued", dict(id=str(index), patient=str(index//3)), {}) for index in range(12)]
        chosen = sample(records, 3)
        self.assertEqual(chosen, sample(list(reversed(records)), 3))
        self.assertEqual(len({item[1]["patient"] for item in chosen}), 3)

    def test_refuses_wrong_training_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            folder = run/"crossfit"/"1"
            save(folder/"predictions.json", [])
            save(folder/"provenance.json", dict(folds=[1, 3], predictions=hash_file(folder/"predictions.json")))
            rows = [dict(cohort="busbra", id=str(fold), patient=str(fold), fold=fold) for fold in range(1, 6)]
            with self.assertRaisesRegex(ValueError, "provenance"):
                cases(run, rows)

    def test_does_not_substitute_pilot_or_overwrite_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [dict(cohort="busbra", id=str(fold), patient=str(fold), fold=fold) for fold in range(1, 6)]
            save(run/"manifest.json", dict(demo=False, rows=rows+[dict(cohort="uclm")]))
            save(run/"pilot"/"predictions.json", [])
            with self.assertRaisesRegex(ValueError, "not ready"):
                generate(run)
            self.assertFalse((run/"failure-review").exists())
            (run/"failure-review").mkdir()
            with self.assertRaisesRegex(FileExistsError, "preserved"):
                generate(run)

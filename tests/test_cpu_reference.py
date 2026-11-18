import unittest
from unittest.mock import patch

import numpy as np

from scripts.cpu_reference import development, evaluate


class PositionTests(unittest.TestCase):
    def test_only_other_folds_supply_template(self):
        rows = [dict(cohort="busbra", patient=str(fold), fold=fold) for fold in range(1, 6)]
        rows.append(dict(cohort="uclm"))
        def mask(row):
            self.assertIn(row["fold"], (1, 2, 3))
            result = np.zeros((128, 128), dtype=bool)
            result[row["fold"], :] = True
            return result
        with patch("scripts.cpu_reference.native_mask", side_effect=mask):
            result = evaluate(rows)
        self.assertEqual(result["patients"], 3)
        self.assertEqual(result["patient_macro_dice"], 0)
        self.assertTrue(all(fold["missed_scans"] == 1 for fold in result["folds"]))

    def test_detects_overlap_with_reserved_source_fold(self):
        with self.assertRaisesRegex(ValueError, "crosses folds"):
            development([dict(cohort="busbra", patient="same", fold=fold) for fold in (1, 4)])

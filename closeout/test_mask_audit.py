import unittest

import numpy as np

from closeout.mask_audit import measure


class PixelAuditTests(unittest.TestCase):
    def test_native_edit_accounting(self):
        first = np.zeros((8,8),dtype=bool)
        first[1:4,1:4] = True
        target = first.copy()
        target[1,1], target[4,4] = False, True
        second = first.copy()
        second[1,1], second[2,2], second[4,4], second[0,0] = False, False, True, True
        result = measure(first,second,target)
        for key in ("lesion_recovered","lesion_erased","false_positive_removed","false_positive_added"):
            self.assertAlmostEqual(result[key],1/9)
        self.assertEqual(result["first"],result["second"])
        self.assertEqual(result["outside_fraction"],0)

    def test_crop_clipping_is_not_refiner_erasure(self):
        first = np.zeros((10,10),dtype=bool)
        first[1,1] = True
        target = first.copy()
        target[8,8] = True
        result = measure(first,first.copy(),target)
        self.assertEqual(result["outside_fraction"],.5)
        self.assertEqual(result["lesion_erased"],0)
        self.assertEqual(result["headroom"],0)
        with self.assertRaisesRegex(ValueError,"outside"):
            measure(first,target,target)

    def test_empty_first_pass_uses_full_frame(self):
        target = np.eye(4,dtype=bool)
        result = measure(np.zeros_like(target),target,target)
        self.assertEqual(result["ceiling"],1)
        self.assertEqual(result["second"],1)
        self.assertEqual(result["lesion_recovered"],1)

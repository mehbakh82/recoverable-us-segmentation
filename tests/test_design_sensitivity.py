import unittest

from scripts.design_sensitivity import scenario


class DesignSensitivityTests(unittest.TestCase):
    def test_identical_decisions_cannot_differ(self):
        result = scenario(51,.10,20,40)
        self.assertEqual(result["trial_delta_range_95"],[0.,0.])
        self.assertEqual(result["fraction_lower_bound_positive"],0.)

    def test_expected_advantage_and_reproducibility(self):
        result = scenario(46,.10,20,40)
        self.assertAlmostEqual(result["expected_delta"],5*.10/256)
        self.assertEqual(result,scenario(46,.10,20,40))
        self.assertEqual(scenario(31,0.,20,40)["expected_delta"],0.)

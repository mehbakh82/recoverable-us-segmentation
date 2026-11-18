import itertools
import random
import unittest
from fractions import Fraction

from rus.routing import gate, selected


def endpoint(rows, allocation):
    totals = {}
    for row in rows:
        if row["positive"]:
            values = totals.setdefault(row["patient"], [])
            value = row["second"] if row["id"] in allocation else row["first"]
            values.append(Fraction(str(value)))
    return sum(sum(values)/len(values) for values in totals.values())/len(totals)


def cohort(examples):
    return [dict(id=str(index), patient=patient, positive=positive, first=first, second=second,
                 features={"entropy": index})
            for index, (patient, positive, first, second) in enumerate(examples)]


class ExhaustiveOracleTests(unittest.TestCase):
    def test_every_subset_and_budget(self):
        scenarios = [
            [(0, True, .5, 1.), (0, True, .5, .5), (0, True, .5, .5),
             (1, True, .5, .75), (1, False, 0., 1.), (2, False, 1., 0.)],
            [(0, True, .75, .25), (0, True, .75, .5), (1, True, 1., .5),
             (1, False, 1., 0.), (2, True, 1., .75)],
            [(0, True, .5, .75), (1, True, .5, .75), (2, True, .5, .75),
             (3, True, .25, .25), (4, False, 0., 0.)],
        ]
        rng = random.Random(2411)
        for size in range(5, 11):
            for _ in range(5):
                scenarios.append([(rng.randrange(4), index == 0 or rng.random() < .75,
                                   rng.randrange(5)/4, rng.randrange(5)/4) for index in range(size)])
        for examples in scenarios:
            rows = cohort(examples)
            ids = [row["id"] for row in rows]
            for count in range(len(rows)+1):
                with self.subTest(size=len(rows), retries=count, examples=examples):
                    best = max(endpoint(rows, allocation) for allocation in itertools.combinations(ids, count))
                    chosen = selected(rows, "oracle", Fraction(count,len(rows)))
                    self.assertEqual(len(chosen), count)
                    self.assertEqual(endpoint(rows, chosen), best)
                    if count == len(rows)//5:
                        pilot = gate(rows)
                        self.assertAlmostEqual(pilot["scores"]["oracle"], float(best))
                        entropy = endpoint(rows, ids[-count:]) if count else endpoint(rows, ())
                        self.assertEqual(pilot["passed"], best-entropy >= Fraction(1,100) and best > endpoint(rows, ()))

    def test_inclusive_headroom_and_strict_improvement(self):
        for first in (.1,.4,.55):
            for change, expected in ((.05-1e-9, False), (.05, True), (.05+1e-9, True)):
                rows = cohort([(index, True, first, first+(change if index == 0 else 0.)) for index in range(5)])
                self.assertEqual(gate(rows)["passed"], expected)
        rows = cohort([(index, True, .5, .4 if index == 4 else .5) for index in range(5)])
        self.assertGreater(gate(rows)["oracle_minus_entropy"], .01)
        self.assertFalse(gate(rows)["passed"])

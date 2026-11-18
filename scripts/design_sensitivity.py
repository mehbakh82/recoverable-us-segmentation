"""Simulated fixed-budget routing contrasts. No clinical inputs or fitted models."""
import argparse
from pathlib import Path

import numpy as np

from rus.shared import save


def scenario(common, advantage, trials=1000, bootstraps=2000):
    scans, positives, retries = 256, 252, 51
    magnitude = .20
    if not 0 <= common <= retries or abs(advantage) > magnitude or min(trials, bootstraps) < 2:
        raise ValueError("Invalid overlap, advantage, or repetition count")
    swaps = retries - common
    rng = np.random.default_rng([2411, common, round((advantage+magnitude)*10000)])
    normals = rng.multivariate_hypergeometric([swaps, swaps, scans-2*swaps], scans-positives, size=trials)
    gain_only = swaps - normals[:, 0]
    difficulty_only = swaps - normals[:, 1]
    helped_gain = rng.binomial(gain_only, (.5 + advantage/(2*magnitude)))
    helped_difficulty = rng.binomial(difficulty_only, .5)
    positive_diff = helped_gain + difficulty_only - helped_difficulty
    negative_diff = gain_only - helped_gain + helped_difficulty
    deltas = magnitude * (positive_diff-negative_diff)/positives
    excludes_zero_above = 0
    for plus, minus in zip(positive_diff, negative_diff):
        proportions = np.array([plus, minus, positives-plus-minus])/positives
        draws = rng.multinomial(positives, proportions, size=bootstraps)
        sampled = magnitude*(draws[:, 0]-draws[:, 1])/positives
        excludes_zero_above += int(np.quantile(sampled, .025) > 0.)
    frequency = excludes_zero_above/trials
    return dict(shared_retries=common, swaps=swaps, exclusive_mean_gain_advantage=advantage,
                expected_delta=swaps*advantage/scans, simulated_mean_delta=float(deltas.mean()),
                trial_delta_range_95=np.quantile(deltas,[.025,.975]).tolist(),
                fraction_lower_bound_positive=frequency, monte_carlo_se=float(np.sqrt(frequency*(1-frequency)/trials)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = dict(scans=256, positive_patients=252, retries_per_policy=51, seed=2411,
                    individual_retry_change_magnitude=.20, trials=1000, bootstraps=2000)
    results = [scenario(common, advantage) for common in (51,46,31,0) for advantage in (0.,.05,.10)]
    save(args.output, dict(kind="assumption-based simulation; not clinical evidence", settings=settings, scenarios=results))
    print("shared  swaps  exclusive advantage  expected delta  fraction with lower CI > 0")
    for result in results:
        print(f'{result["shared_retries"]:6} {result["swaps"]:6} {result["exclusive_mean_gain_advantage"]:20.2f} '
              f'{result["expected_delta"]:+15.4f} {result["fraction_lower_bound_positive"]:27.1%}')


if __name__ == "__main__":
    main()

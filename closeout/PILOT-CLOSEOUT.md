# Breast ultrasound refinement: fold-4 pilot closeout

**Decision, 18 November 2026: stop before training a router.** On the BUS-BRA pilot fold (365 lesion-positive scans, 214 patients), a second pass helped when allocated by first-pass entropy. The best possible allocation did better, but its advantage over entropy was **0.009073** patient-macro Dice, below the prespecified **0.010000** required to continue.

| Policy | Retries | Patient-macro Dice | Change from first pass |
|---|---:|---:|---:|
| Never retry | 0 | 0.858080 | — |
| Highest first-pass entropy | 73 | 0.868296 | +0.010217 |
| Oracle, using reference masks | 73 | 0.877370 | +0.019290 |

## What explains the gap?

The policies agreed on **35 of their 73 retries (47.9%)**. They did not simply pick the same cases. The common cases accounted for all of entropy routing's net benefit: every one improved. Entropy's other 38 retries slightly reduced the cohort score in aggregate.

| Selected scans | Improved | Worsened | Unchanged | Contribution to whole-fold Dice above no retry |
|---|---:|---:|---:|---:|
| Both policies (35) | 35 | 0 | 0 | +0.012320 |
| Entropy only (38) | 18 | 19 | 1 | −0.002104 |
| Oracle only (38) | 38 | 0 | 0 | +0.006969 |

Replacing the entropy-only group with the oracle-only group accounts for the entire **+0.009073** difference. Across all 73 entropy retries, **53 improved, 19 worsened and one was unchanged**. None of those retries changed a zero-overlap lesion mask to nonzero overlap or vice versa. Thus the observed Dice improvement was not a recovery of completely missed lesions. “Improved” and “worsened” mean any signed change in mask Dice, not clinically established benefit or harm.

Counts in the table are images. Score contributions are not average changes *within* a selected group: each scan's change is weighted by the inverse of its patient's number of positive scans and averaged over all 214 patients. This is why the contributions can be added to recover the policy scores.

## Interpretation and independent follow-up

Entropy identified a useful set of refinements but did not reliably distinguish helpful from harmful retries among its exclusive selections. With these two fixed segmenters, this population and this 20% budget, even perfect hindsight could not achieve the prespecified incremental gain. This is an investment decision for this study, not a claim of statistical equivalence or that recoverability is unlearnable. No learned-router effectiveness, external generalization or clinical utility was established.

**Recommendation:** retain the negative pilot as the result. If a new study is proposed, specify what would change about the refiner or endpoint and why before selecting a new evaluation cohort. Do not lower the gate or use the unused test cohorts to rescue this hypothesis.

The subsequent [post-hoc pixel-change audit](MASK-AUDIT.md) examines the 73 entropy-selected retries. It did not change the stop decision. The [machine-readable pilot summary](pilot.aggregate.json) records the counts and scores above.

## Evidence and reproduction

The H100 ran both fixed models for 50 epochs each. The CPU closeout matched the original local pre-pilot archive (SHA-256 `a3669d405064959e29aa013f0d2d3ec90d7d47247fcab893f43c733426346de0`), verified the portable manifest, training/held-out patient separation, saved checkpoint hashes and prediction digest, and reproduced the recorded gate. The remote manifest's raw hash differs because paths moved; its portable hash matches. Six copied aggregate files were checksum-verified; the separate startup verification receipt matched the receipt in the gate. No scan-level records, masks or weights were copied for this closeout.

To reproduce on the study host from the frozen project root, provide [the closeout script](pilot_summary.py) on standard input: `CUDA_VISIBLE_DEVICES="" venv/bin/python - runs/msc < /path/to/pilot_summary.py`. It reads only metadata, development checkpoint hashes/ledgers, and saved fold-4 outcomes, returning aggregate JSON. Fold 5 and both external cohorts were not scored. The overlap analysis is descriptive and was not used to redefine the gate.

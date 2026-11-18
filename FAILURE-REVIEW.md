# Development mask review worksheet

**Scope:** BUS-BRA folds 1–3, using predictions from models trained on the other two development folds. Never substitute pilot, internal-test or external predictions. This is a descriptive review of segmentation geometry, not a clinical diagnosis or another model-selection gate.

## What to record for each scan

| Field | Reviewer entry |
|---|---|
| Case alias / reviewer / date | |
| First pass | empty / nonempty but no overlap / partial overlap / close contour |
| Second pass | empty / nonempty but no overlap / partial overlap / close contour |
| Main visible change | undercoverage / overcoverage / displacement / disconnected component / no visible difference / unclear |
| Evidence | One sentence locating the difference relative to the reference; e.g. “The second contour omits the reference's lower edge.” |
| Confidence | clear / uncertain / not assessable |

An **empty mask** has no foreground pixels. A **miss** has no overlap with the lesion reference and can be either empty or nonempty. A **rescue** gains some overlap after a miss; it need not be a good final contour. A **loss of overlap** is the converse. For cases with overlap in both passes, compare native-resolution Dice to distinguish improvements, harms and ties.

## Sampling and interpretation

The generator reports counts for **every eligible development scan**, then draws up to four distinct patients per outcome category in a deterministic order. One patient can appear in more than one category. The selected panels deliberately overrepresent unusual events: do not use them to estimate prevalence or patient-level effectiveness. Record what is visible; do not infer pathology, acquisition artifacts or a reason for failure without evidence.

Panels show the same frame with the reference outline alone, then with the first and second predictions. Outlines are green for reference, magenta for prediction and white where the boundaries coincide. Use the printed empty/miss flags if an outline is absent. Display resizing is not used for scoring. Open the generated local Markdown file, fill in its case cards and retain it privately. Do not put images, identifiers or annotated case cards on GitHub.

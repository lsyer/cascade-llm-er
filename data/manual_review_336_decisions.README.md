# Manual Review Decisions (336 cases)

Human adjudication decisions correcting the MINEC ground truth. Two sections:
unresolved cases and GLM-5.2/Claude Opus disagreements. Format: `N. p_XXXX，label`
(same / different / unknown). 336 entries, no duplicates:
same 232, different 56, unknown 48.

## Verified relation to minec_ground_truth_v2.json (2026-11-04 audit)

- 331 entries → `label_source = "manual_mark_txt"`; final_label matches the
  txt decision in all 331 cases.
- 5 entries (p_311, p_430, p_577, p_579, p_825; all txt=same) also fall in the
  300-pair blind annotation sample; the blind label (different) superseded the
  review decision → `label_source = "human_adjudication_300"`.
- The 48 `unknown` decisions correspond exactly to the 48 unknowns retained in
  the ground truth.

## Status

Audit/provenance artifact — the original human input. The GT JSON is the
canonical machine-readable version; this file is not a reproducible experiment
input.

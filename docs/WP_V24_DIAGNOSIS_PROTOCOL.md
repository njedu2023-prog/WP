# WP V24 Failure-Attribution Protocol

## Purpose

V24 is closed and rejected. This diagnostic reads its immutable nested
out-of-sample evidence only to determine why the wider top-five opportunity
set failed and whether the existing causal feature family contains any stable
within-slot ranking information.

It cannot authorize a successor, shadow run, production model, threshold, or
subgroup.

## Fixed Diagnostics

- Verify selected candidate count and identity uniqueness against the frozen
  V24 summary.
- Measure positive, margin, and severe-loss probability discrimination.
- Measure expected-return and economic-score rank correlation with realized
  net return.
- Measure every model output's same-day same-slot ranking IC and
  highest-minus-lowest desirability return spread.
- For every active V24 causal feature, calculate same-day same-slot
  cross-sectional Spearman IC and highest-minus-lowest feature return spread.
- Aggregate cross-sections by trade date before inference.
- Report year-level sign stability.
- Apply Benjamini-Hochberg correction across the causal feature family.
- Attribute selected gains and losses by year, signal slot, source fold,
  source rank, and prior appearance count.
- Report fold threshold and candidate-frequency drift.

An exploratory causal feature is labelled stable only when it has at least 100
cross-sections, mean daily-slot IC at least 0.05, BH q-value at most 0.10,
average top-minus-bottom return spread at least +0.20 percentage points,
positive IC in at least three years, minimum yearly IC above -0.02, and
minimum yearly spread above -0.25 percentage points.

This label is a hypothesis-generation aid, not confirmation. Any V25 policy
must be separately specified and frozen before another evaluation.

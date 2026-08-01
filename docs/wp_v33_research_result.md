# WP V33 Limit-Industry Ecology Research Result

## Decision

V33 is rejected. It is not authorized for production or shadow deployment.
The frozen policy selected no candidate in 725 nested out-of-sample trading
days. Threshold repair or reuse of V33 as a production candidate generator is
forbidden.

## Immutable Evidence

- Repository: `njedu2023-prog/WP`
- Valid recovery workflow run: `30675663679`
- Source commit: `e30f8b4a4f116f0dc8c839e7d7b23c939fe6f7a3`
- Artifact: `wp-v33-limit-industry-research-30675663679`
- Artifact ID: `8810440531`
- Artifact digest:
  `sha256:1405e4fbb003b86196b3428d80506ea5f147ee19fdfa20ede520a8ab90029ebf`
- Failed pre-result workflow run: `30675048976`
- Recovery audit: `docs/wp_v33_research_recovery.md`

## Nested Out-Of-Sample Result

- Evaluation days: `725`
- Model-covered days: `725`
- Source candidates: `31,955`
- Joined candidates: `31,955`
- Selected candidates: `0`
- Candidate days: `0`
- Candidate-day rate: `0.00%`
- Final 42-day calibration eligible days: `0`
- Final score threshold: not defined because no row passed the frozen
  absolute gates

No economic return statistic can be claimed from an empty selected sample.

## Rank Diagnostics

- Evaluable same-slot groups: `2,950`
- Evaluable rank days: `615`
- Mean within-slot rank IC: `0.008482`
- Mean top-minus-bottom return spread: `+0.177940%`
- Clustered spread lower bound: `+0.037526%`
- Positive spread years: `3 / 4`

The ecology feature family contains weak ordering information, but it misses
both preregistered rank gates: rank IC must be at least `0.05` and top-minus-
bottom spread must be at least `+0.20%`.

## Integrity

- Temporal integrity: passed
- Source integrity: passed
- Data integrity: passed
- Candidate identity audit: passed
- Selected-outcome audit: passed
- Historical economic gates: failed
- Historical rank gates: failed
- Production authorized: `false`

## Interpretation

This is not a data-pipeline failure. The V33 industry-limit ecology variables
do not provide enough independent candidate-level information under the frozen
execution and risk contract. A next attempt must introduce a genuinely new,
causal information family and a new preregistered nested-OOS protocol. It may
not repair V33 by relaxing absolute gates or choosing a threshold after seeing
these outcomes.

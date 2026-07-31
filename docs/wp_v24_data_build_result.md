# WP V24 Point-in-Time Data Build Result

## Decision

The preregistered V24 top-five point-in-time dataset passed every construction
and coverage gate. The frozen V24 nested out-of-sample cross-section research
is authorized. This result does not authorize a strategy, shadow deployment,
or production use.

The candidate identity set was selected without reading profit outcomes. It
contains at most five immutable V9 broad-recall candidates per market slot and
only causal information available by each signal slot.

## Immutable Evidence

- Protocol and implementation commit:
  `78aa6419b378eef7d9528db5ba24b0304e4497ab`
- Immutable V9 source run: `30600193544`
- V24 data build run: `30635569735`
- V24 data build job: `91172189423`
- Artifact: `8798280716`
- Artifact name: `wp-v24-point-in-time-data-30635569735`
- Artifact digest:
  `sha256:da23a74919e920f42d9449b90971b178a205338194897b325c22208565362e2b`
- Artifact size: `33,906,618` bytes
- Outcome-blind candidate rows: `31,955`
- Source trade dates: `913`
- Required stock-month pairs: `9,669`
- `v24_model_research_authorized`: `true`

## Query Coverage

| Data family | Required units | Ready rows | Coverage | Query failures |
| --- | ---: | ---: | ---: | ---: |
| Historical one-minute bars | 9,669 stock-month pairs | 31,955 | 100.0000% | 0 |
| Same-day completed opening auction | 913 dates | 31,955 | 100.0000% | 0 |
| Immediately previous-day L2 money flow | 913 dates | 31,950 | 99.9844% | 0 |
| Complete causal feature rows | 31,955 candidates | 31,950 | 99.9844% | 0 |

The preregistered minimum complete-dataset coverage was `98%`. Five candidate
rows lack previous-day money-flow coverage and remain in the immutable source
index; they are marked incomplete rather than being removed according to
outcomes.

## Integrity Controls

- The top-five candidate identity set was fixed before any profit outcome was
  read.
- Each slot contains no more than five source candidates and no duplicate
  candidate identity.
- One-minute features use observations at or before each signal slot.
- Opening-auction fields use only the completed same-day auction.
- Money-flow fields use only the immediately preceding A-share trading day.
- Query failures are recorded by data family and all three failure counts are
  zero.
- The manifest preserves all `913` source trade dates for candidate-frequency
  denominators.
- Missing or inconsistent outcomes for any selected historical signal are a
  hard research-integrity failure and cannot be silently removed.

## Authorized Next Step

Run exactly one frozen V24 nested out-of-sample study using the preregistered
model, policy, execution contract, calendar denominators, and acceptance
gates. No post-result threshold search is allowed.

Passing all historical gates can authorize only the required 150 future
A-share trading-day shadow run. It cannot authorize production.

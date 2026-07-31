# WP V23 Point-in-Time Data Build Result

## Decision

The preregistered V23 point-in-time dataset passed its construction and
coverage gates. The frozen V23 nested out-of-sample model research is
authorized. This result does not authorize a strategy, shadow deployment, or
production use.

The dataset identity set was selected without reading profit outcomes. It
contains the immutable V9 market-slot leaders and only causal information
available by each signal slot.

## Immutable Evidence

- Immutable source run: `30600193544`
- Initial cache build run: `30628882889`
- Initial cache build job: `91150407356`
- Final full-calendar build run: `30630271759`
- Final full-calendar build job: `91161279737`
- Artifact: `8793961080`
- Artifact digest:
  `sha256:52420a8d88477887d48a2499db246f45c0ecb6887a3de1ed67d95f003a18af99`
- Source leaders: `6,338`
- Source trade dates: `908`
- `profit_outcomes_read`: `false`
- `v23_model_research_authorized`: `true`

The final run restored the immutable source cache and the complete V23 query
cache from run `30628882889`, then regenerated the manifest using the full SSE
open-date calendar configured from `20210726` through `20260724`.

## Query Coverage

| Data family | Required units | Ready rows | Coverage | Query failures |
| --- | ---: | ---: | ---: | ---: |
| Historical one-minute bars | 1,753 stock-month pairs | 6,338 | 100.0000% | 0 |
| Same-day completed opening auction | 908 dates | 6,338 | 100.0000% | 0 |
| Immediately previous-day L2 money flow | 908 dates | 6,330 | 99.8738% | 0 |
| Complete causal feature rows | 6,338 leaders | 6,330 | 99.8738% | 0 |

The preregistered minimum complete-dataset coverage was `98%`. Eight leader
rows lack previous-day money-flow coverage and remain in the immutable source
index; they are marked incomplete rather than being removed according to
outcomes.

## Integrity Controls

- The leader identity set was fixed before any profit outcome was read.
- One-minute features use observations at or before each signal slot.
- Opening-auction fields use only the completed same-day auction.
- Money-flow fields use only the immediately preceding A-share trading day.
- Query failures are recorded by data family and all three failure counts are
  zero.
- The manifest stores the full ordered SSE open-date calendar so no-signal
  dates remain in the candidate-frequency denominator.
- Missing or inconsistent outcomes for any selected historical signal are a
  hard research-integrity failure; they cannot be silently removed from
  evidence.

## Authorized Next Step

Run exactly one frozen V23 nested out-of-sample study using the preregistered
model, policy, execution contract, calendar denominators, and acceptance
gates. No post-result threshold search is allowed.

Passing historical gates can authorize only the required 150 future A-share
trading-day shadow run. It cannot authorize production.

# WP V32 A-Share Public-Event Data Probe Result

## Decision

V32 passed its frozen outcome-blind data-feasibility gate.

This authorizes only a full three-year point-in-time public-event feature
build. It does not establish predictive value and does not authorize model
selection, shadow evaluation, public candidates, or production use.

## Immutable Evidence

- Frozen implementation commit:
  `56ea3983e747c8fe6696ab3df2789df03bd601d3`
- Workflow run: `30664863503`
- Workflow job: `91269369912`
- Artifact: `wp-v32-a-share-event-data-probe-30664863503`
- Artifact ID: `8806551194`
- Artifact digest:
  `sha256:4c73fcea3c2c209c985bfda385cb655a1279674766d2eabb21f9d9849e75f815`
- Immutable V31 raw event source run: `30663984930`
- Immutable V24 outcome-blind candidate source run: `30635569735`

## Probe Result

- Probe dates: `8`
- Required prior trading dates: `40`
- Immutable source cache files: `240`
- Candidate identities: `121`
- Matched candidate identities: `21`
- Matched target dates: `7 / 8`
- Full backfill authorized: `true`
- Next gate: `full_three_year_outcome_blind_public_event_build`

## Source Results

| Source | Raw rows | Retained A-share rows | Excluded rows | Candidate matches | Target dates | Admitted |
|---|---:|---:|---:|---:|---:|---|
| `forecast` | 346 | 346 | 0 | 3 | 2 | yes |
| `express` | 111 | 111 | 0 | 0 | 0 | no |
| `repurchase` | 976 | 973 | 3 | 3 | 3 | yes |
| `stk_holdertrade` | 1,631 | 1,631 | 0 | 0 | 0 | no |
| `share_float` | 211,619 | 211,584 | 35 | 3 | 3 | yes |
| `block_trade` | 6,225 | 5,259 | 966 | 14 | 6 | yes |

All 240 immutable cache files passed schema, requested-date, duplicate, and
A-share universe-normalization contracts. Four sources passed the unchanged
per-source admission gates.

## Frozen Conclusion

The lagged public-event family has sufficient candidate coverage for a full
outcome-blind build. V32 has not read candidate returns and makes no claim
that these events improve the fixed T+1 close net-return objective.

The full build must retain only the four admitted sources, the five prior
trading-day window, the immutable V24 candidate identities, and exact parity
with the probe identities. Model research remains forbidden until that build
passes its own frozen data contract.

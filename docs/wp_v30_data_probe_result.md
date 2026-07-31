# WP V30 Limit-Event Data Probe Result

## Decision

V30 is closed at the frozen data-feasibility gate.

It is not authorized for a three-year backfill, model research, shadow
evaluation, or production use. No profit outcome was read during source
selection or probing.

## Immutable Evidence

- Frozen implementation commit:
  `454e8e1c5ad9d85f68ca07d8863b9c73be54df6a`
- Final diagnostic-only commit:
  `dc1a2203008c8b3953a5b731a88d821580278feb`
- Workflow run: `30662958173`
- Workflow job: `91263234703`
- Artifact: `wp-v30-limit-event-data-probe-30662958173`
- Artifact ID: `8805887500`
- Artifact digest:
  `sha256:b51694a02851870abfde0a571d10c8e9b36b69539ea43228f1114ce5bbae7132`
- Immutable V24 outcome-blind candidate source run: `30635569735`

The final workflow conclusion is intentionally `failure`: the probe raises
after writing evidence whenever any frozen gate fails.

## Probe Result

- Probe dates: `8`
- Current/previous date-category queries: `48`
- Query contract passed: `false`
- Date projections passed: `8 / 8`
- Point-in-time market-context rows: `56 / 56`
- Sample candidate rows: `280`
- Candidate rows with a causal limit touch: `0`
- Full backfill authorized: `false`
- Next gate: `close_v30_data_direction`

All eight dates produced complete seven-slot market projections. By 14:50,
the reconstructed market had between 38 and 163 limit-up touches per probe
date, so the market-level event source was not empty.

## Failure Attribution

### Candidate-level coverage

The exact `(trade_date, ts_code)` identity join was checked independently of
signal time. It returned zero event-matched candidate rows on every probe
date:

| Trade date | Candidate rows | Unique candidate codes | Event-matched rows |
|---|---:|---:|---:|
| 2023-08-25 | 35 | 13 | 0 |
| 2023-12-29 | 35 | 18 | 0 |
| 2024-03-15 | 35 | 15 | 0 |
| 2024-09-27 | 35 | 13 | 0 |
| 2025-01-15 | 35 | 13 | 0 |
| 2025-07-23 | 35 | 15 | 0 |
| 2026-01-15 | 35 | 15 | 0 |
| 2026-07-23 | 35 | 19 | 0 |

This rules out a signal-time comparison bug: the sampled V24 candidates do
not belong to the same-day limit-up, failed-limit, or limit-down event set at
all.

### Timestamp contract

Eleven of 48 queries failed the frozen `09:25:00-15:00:00` timestamp
contract. All supplied timestamps parsed correctly. The violations were
source records at `15:00:02-15:00:04`, usually in `last_time` or
`open_time`; one limit-up timestamp was `15:00:04`.

These few seconds do not explain the zero candidate identity overlap.
Changing the timestamp boundary after observing the probe would also violate
the frozen protocol.

## Frozen Conclusion

The source can reconstruct a dense market-level limit-event tape, but the
frozen candidate-level mechanism has no sampled coverage. V30 therefore
cannot test whether a candidate's own prior limit event improves the
executable T-to-T+1 net-return objective.

The following are prohibited under V30:

- deleting the candidate-overlap gate;
- changing the probe dates or candidate source;
- accepting post-close timestamp tolerance after observing the failure;
- treating market-event availability as evidence of predictive value;
- reading returns to choose a rescue rule;
- reusing V30 as a confirmatory test.

Any market-only limit-event hypothesis must be registered as a separate,
independent version with a new outcome-blind protocol and fresh evaluation
budget.

# WP V33 Three-Year Limit-Industry Ecology Data Protocol

## Decision

The eight-date, outcome-blind V33 feasibility probe passed every frozen
coverage, time-causality, membership, and V30 parity gate. This protocol
therefore authorizes one full three-year point-in-time data build. It does not
authorize model training, threshold search, production use, or any claim of
profitability.

## Immutable Sources

- V24 candidate identities and folds: Actions run `30635569735`.
- V28 point-in-time SW2021 L2/L3 membership: Actions run `30656696310`.
- V33 outcome-blind probe: Actions run `30671024383`.
- Tushare `kpl_list`, queried once per required trade date and per fixed tag:
  `涨停`, `炸板`, and `跌停`.

Every source artifact must pass its schema, outcome-blindness, row-count, and
SHA-256 contract before the build can continue.

## Time Contract

- Target slots are exactly `14:20`, `14:25`, `14:30`, `14:35`, `14:40`,
  `14:45`, and `14:50`.
- Current-day features may use only first limit touch, first open-board, or
  first limit-down timestamps at or before the candidate slot.
- Current-day events after `14:50` are ignored.
- Previous-day features use only the completed immediately preceding A-share
  trading day.
- The candidate stock is excluded from every same-industry peer aggregate.
- No T+1 outcome, future close, future event state, or profitability field may
  be read.

## Full-Build Coverage Gates

The full build is accepted only if all gates pass without threshold changes:

1. Candidate identities exactly match all `31,955` immutable V24 rows.
2. Every required date/tag query passes schema and timestamp validation, with
   zero failed queries.
3. Every required date produces exactly seven unique market projections.
4. Candidate L2/L3 membership coverage is at least `98%`.
5. Current-day and previous-day event-stock L2/L3 membership coverage are each
   at least `90%`.
6. Same-L2 limit-event activity covers at least `20%` of candidate rows and
   at least `70%` of target trading days.
7. Same-L3 limit-event activity covers at least `10%` of candidate rows and
   at least `50%` of target trading days.
8. Every numeric feature is finite and complete.
9. The full build reproduces all 280 probe candidate identities and all frozen
   V33 numeric features within absolute tolerance `1e-12`.
10. Forbidden end-state fields and raw future outcomes are absent.

Failure closes the V33 data direction. It does not permit relaxing a gate.

## Outputs

- `wp_v33_outcome_blind_candidate_index.parquet`
- `wp_v33_limit_industry_ecology_features.parquet`
- `wp_v33_market_projection.parquet`
- `wp_v33_source_date_audit.json`
- `wp_v33_limit_industry_ecology_data_manifest.json`

Only a passing manifest may authorize a separately preregistered V33 nested
out-of-sample model protocol.

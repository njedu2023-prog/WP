# WP V31 Public-Event Data Feasibility Protocol

## Status

This protocol is frozen before any V31 profit outcome is read.

The probe is outcome-blind. A pass authorizes only a full three-year
point-in-time data build. It does not authorize model research, shadow
evaluation, public candidates, or production use.

## Independent Hypothesis

V24-V30 already tested causal price/volume microstructure, positioning,
abnormal-trading disclosures, attention-source availability, fine-industry
peer confirmation, and same-day limit-event state.

V31 tests a different mechanism: whether company-specific information made
public before the signal date, plus prior off-book transactions, provides
incremental context for ranking otherwise eligible tail-session candidates.

The six frozen sources are:

1. `forecast`: earnings forecasts;
2. `express`: earnings express reports;
3. `repurchase`: share-repurchase announcements;
4. `stk_holdertrade`: major-holder and executive share changes;
5. `share_float`: restricted-share release announcements;
6. `block_trade`: block trades.

## Causal Contract

For every target signal date, V31 may use only records dated in the five
immediately preceding A-share trading days.

- Announcement sources use `ann_date`.
- `block_trade` uses `trade_date`.
- Same-day records are forbidden because the standard historical interfaces
  do not provide a uniformly reproducible intraday publication timestamp.
- Calendar dates that are not in the frozen five-trading-day lookback are
  forbidden.
- Empty source-date responses are valid zero-event observations.
- Revised fields, later announcement versions, T+1 fields, candidate returns,
  and truth outcomes are forbidden during this probe.

The five-day horizon is fixed before source access. It cannot be expanded
after observing candidate overlap.

## Immutable Candidate Source

- V24 outcome-blind point-in-time data run: `30635569735`;
- exact V24 candidate identities only;
- probe dates:
  - `2023-08-25`
  - `2023-12-29`
  - `2024-03-15`
  - `2024-09-27`
  - `2025-01-15`
  - `2025-07-23`
  - `2026-01-15`
  - `2026-07-23`

The dates and candidates cannot be replaced based on data density.

## Per-Source Admission Gate

Each source is evaluated independently and is admitted only if:

1. every required source-date request succeeds;
2. every response has the frozen schema;
3. all nonempty response dates equal the requested prior date;
4. every code is a valid A-share code;
5. no exact duplicate record exists;
6. at least four requested dates are nonempty;
7. at least two unique target-date/candidate-code identities match;
8. those matches span at least two probe target dates.

Failure of an optional source does not invalidate another source that passes
its own frozen gate.

## Family Gate

The V31 family passes only if:

1. at least two sources pass their per-source gates;
2. their union matches at least ten unique target-date/candidate-code
   identities;
3. union matches span at least five of the eight probe target dates;
4. every sampled candidate identity receives complete source-presence flags;
5. every contributing source date is strictly earlier than its target date
   and belongs to the frozen five-day lookback;
6. no profit outcome is read or written.

## Decision Rule

```text
if all family gates pass:
    full_backfill_authorized = true
    next_gate = full_three_year_outcome_blind_public_event_build
else:
    full_backfill_authorized = false
    next_gate = close_v31_data_direction
```

A later model protocol, if authorized, must be frozen separately. It must
retain the established executable next-five-minute-close entry, fixed T+1
close exit, transaction costs, same-day same-slot ranking gates, nested
out-of-sample evaluation, and at least 150 untouched future A-share trading
days before production.


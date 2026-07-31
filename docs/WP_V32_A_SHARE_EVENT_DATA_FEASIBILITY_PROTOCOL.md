# WP V32 A-Share Public-Event Data Feasibility Protocol

## Status

This protocol is frozen before any V32 profit outcome is read.

V32 is a diagnostic successor to V31. A pass authorizes only a full
three-year point-in-time data build. It does not authorize model research,
shadow evaluation, public candidates, or production use.

## Immutable Inputs

- V31 workflow run: `30663984930`;
- V31 artifact ID: `8806378449`;
- V31 artifact digest:
  `sha256:11f60e1b87609524cae09405221a6c096eec9c3319e5b550ba87f69188b8fec4`;
- V24 outcome-blind candidate source run: `30635569735`;
- the same eight probe dates used by V31;
- the same five immediately preceding A-share trading days per target date;
- the same six public-event sources used by V31.

V32 does not re-query or replace the V31 raw source responses.

## Single Corrective Change

The V31 raw-code contract incorrectly rejected a complete source-date
response when any row belonged to a non-A-share security universe.

V32 freezes this normalization rule:

1. validate the full raw response schema and requested date;
2. define an A-share row as a six-digit code in the Shanghai `6xxxxx`,
   Shenzhen `0xxxxx`/`3xxxxx`, or Beijing `4xxxxx`/`8xxxxx`/`9xxxxx`
   namespaces with the matching `.SH`, `.SZ`, or `.BJ` suffix;
3. exclude all other code rows before feature construction and candidate
   joins;
4. record raw rows, excluded non-A-share rows, and retained A-share rows for
   every source-date response;
5. require no exact duplicates among retained A-share rows.

Excluded rows are neither failures nor candidate observations. No other V31
contract is changed.

## Causal Contract

For every target signal date, V32 may use only records dated in the five
immediately preceding A-share trading days.

- Announcement sources use `ann_date`.
- `block_trade` uses `trade_date`.
- Same-day records remain forbidden.
- Dates outside the frozen five-day lookback remain forbidden.
- Empty A-share responses are valid zero-event observations.
- T+1 fields, candidate returns, and truth outcomes are forbidden.

## Per-Source Admission Gate

Each source is admitted only if:

1. all 40 immutable source-date cache files exist and are readable;
2. every raw response has the V31 frozen schema;
3. all nonempty raw response dates equal the requested prior date;
4. universe normalization is fully accounted for;
5. no exact duplicate retained A-share record exists;
6. at least four requested dates retain one or more A-share rows;
7. at least two unique target-date/candidate-code identities match;
8. those matches span at least two probe target dates.

## Family Gate

The family passes only if:

1. at least two sources pass their per-source gates;
2. their union matches at least ten unique target-date/candidate-code
   identities;
3. union matches span at least five of the eight probe target dates;
4. every sampled candidate identity receives complete source-presence flags;
5. every contributing source date is strictly earlier than its target date
   and belongs to the frozen five-day lookback;
6. no profit outcome is read or written.

These are identical to the V31 coverage thresholds.

## Decision Rule

```text
if all family gates pass:
    full_backfill_authorized = true
    next_gate = full_three_year_outcome_blind_public_event_build
else:
    full_backfill_authorized = false
    next_gate = close_v32_data_direction
```

A later model protocol, if authorized, must be frozen separately. It must
retain the executable next-five-minute-close entry, fixed T+1 close exit,
transaction costs, same-day same-slot ranking gates, nested out-of-sample
evaluation, and at least 150 untouched future A-share shadow trading days.

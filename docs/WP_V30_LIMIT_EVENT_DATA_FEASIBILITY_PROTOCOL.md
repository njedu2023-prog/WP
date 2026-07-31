# WP V30 Limit-Event Tape Data Feasibility Protocol

## Status

This protocol is frozen before any V30 return outcome is read.

The probe is outcome-blind. A pass authorizes only a full historical
point-in-time data build. It cannot authorize a model, shadow run, public
candidate, or production use.

## Research Question

V24-V29 established that price-derived features, opening-auction features,
lagged money flow, market breadth, fine-industry peers, holder cost, financing
positioning, and abnormal-trading disclosures do not provide a stable,
economically material same-day same-slot stock-selection mechanism.

V30 tests an independent information family: the discrete tape of limit-up,
failed-limit, and limit-down events that had already occurred by each
14:20-14:50 decision time.

The hypothesis is not that an end-of-day limit list predicts returns. The
hypothesis is that causally observed first-touch and first-open events may
describe:

- whether a candidate had already touched its limit;
- whether that candidate had already opened after touching;
- time elapsed since the first touch or open;
- market-wide limit-touch acceleration;
- market-wide failed-board pressure;
- market-wide limit-down pressure;
- the candidate's fully completed prior-day board state.

## Historical Source

- API: `kpl_list`
- Current-day categories queried independently: `涨停`, `炸板`, `跌停`
- Prior A-share trading-day categories: the same three categories
- Probe dates:
  - `20230825`
  - `20231229`
  - `20240315`
  - `20240927`
  - `20250115`
  - `20250723`
  - `20260115`
  - `20260723`
- Signal slots:
  - `14:20`
  - `14:25`
  - `14:30`
  - `14:35`
  - `14:40`
  - `14:45`
  - `14:50`

The immutable, outcome-blind V24 candidate index and SSE open-date calendar
from run `30635569735` provide candidate identities and previous-trading-day
mapping. No V24 or later return field is read.

## Current-Day Causal Projection

Only event identity and timestamps at or before a signal slot are admissible:

- `ts_code`
- `trade_date`
- first limit-up touch time from `lu_time`
- first open time from `open_time`
- first limit-down touch time from `ld_time`

All event times are discretized to completed five-minute slots to match live
capture resolution.

The union of the end-of-day `涨停` and `炸板` categories may be used only to
recover the set of stocks that touched limit-up. The final category itself is
forbidden as a current-day feature. A future open time is projected as
not-yet-open until its timestamp has passed.

The following current-day fields are explicitly forbidden because they encode
the final state, future events, or end-of-day aggregates:

- `tag`
- `theme`
- `status`
- `lu_desc`
- `last_time`
- `net_change`
- `limit_order`
- `amount`
- `turnover_rate`
- `pct_chg`
- `rt_pct_chg`
- `lu_limit_order`
- `free_float`
- `bid_amount`
- `bid_change`
- `bid_turnover`
- `lu_bid_vol`

## Live Reconstruction Contract

Production cannot depend on the current-day `kpl_list` endpoint being
available before the close.

The live implementation must reconstruct the same five-minute event state from
the already archived all-market `rt_min` session bars plus same-day
`stk_limit` prices:

- a bar whose high reaches the up-limit creates a first-touch event;
- after a first touch, a bar whose low is below the up-limit creates an open
  event;
- a bar whose low reaches the down-limit creates a down-limit event;
- session history is append-only across the anchored five-minute captures.

Historical exact timestamps are therefore discretized to the same completed
five-minute representation before any model research.

## Frozen Probe Gates

The data family passes only if all gates pass:

1. all current-day and previous-day requests for all three categories succeed;
2. every response has the frozen schema and, when nonempty, the requested
   trade date; an empty category is a valid zero-event observation;
3. relevant first-event timestamps have at least `90%` coverage;
4. every nonempty event time is parseable and lies within `09:25-15:00`;
5. no first-open timestamp precedes its first limit-up timestamp;
6. no duplicate stock exists within a requested date-category response;
7. every probe date has at least five causally observed limit-touch or
   limit-down events by `14:50`;
8. all 56 date-slot market-context rows are unique and complete;
9. every sampled V24 candidate receives a complete point-in-time projection;
10. at least one sampled candidate has a causal limit-touch event, proving
    candidate-level rather than market-only overlap;
11. current-day output columns contain no forbidden end-of-day fields;
12. source code and artifacts contain no return outcome.

## Decision Rule

Any failed gate closes this V30 data direction. A pass authorizes a full
three-year outcome-blind event-tape build only.

A later model protocol must be separately frozen, retain the established
executable next-five-minute-close entry and fixed T+1 close exit with costs,
make same-day same-slot rank IC and return spread explicit gates, and require
at least 150 untouched future A-share trading days before production.

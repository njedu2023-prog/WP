# WP V23 Point-in-Time Data Feasibility Protocol

## Purpose

V20, V21, and V22 exhausted three different model structures over the same
five-minute and lagged-daily information base. V23 must not begin with another
model. It begins by proving that a genuinely new, causal data family can be
reproduced across the three-year evaluation span.

This probe reads no candidate returns and makes no trading-policy decision.

## Frozen probes

### Historical one-minute bars

- API: `stk_mins`
- Frequency: `1min`
- Dates: `20230828`, `20240102`, `20250116`, `20260724`
- Symbols: `600000.SH`, `000001.SZ`, `601318.SH`
- Window: 13:55 through 15:00
- Required tail coverage: at least 90% of the expected 60 one-minute bars
  between 14:01 and 15:00 for every symbol-date sample.
- Required quality: unique symbol/timestamp rows and internally consistent
  OHLC fields.
- Control: five-minute samples on the earliest and latest dates.

### Same-day opening auction

- API: `stk_auction_o`
- Dates: `20240102`, `20250116`, `20260724`
- Required coverage: at least 1,000 rows, unique stock-date keys, and all sample
  symbols present.
- Causal use if admitted: the completed T-day 09:30 opening auction, available
  before the 14:20 decision window.

### Lagged daily L2 money flow

- API: `moneyflow`
- Dates: `20230825`, `20231229`, `20250115`, `20260723`
- Required coverage: at least 1,000 rows, unique stock-date keys, and all sample
  symbols present.
- Causal use if admitted: previous-trading-day values only. Same-day completed
  money flow is forbidden because it is unavailable at 14:20.

## Decision

- Historical one-minute bars must pass before any V23 microstructure backfill
  or model research is authorized.
- Auction and lagged money-flow families are independently admitted only if
  their own coverage checks pass.
- A failed family is excluded rather than silently imputed from future or
  alternative data.
- Passing this probe authorizes dataset construction only. It does not
  authorize a strategy, shadow operation, or production.

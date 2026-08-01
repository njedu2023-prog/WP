# WP V39 T-1 Derivatives Data Feasibility Protocol

## Status

This protocol is frozen before any candidate outcome is read. A passing source
family authorizes only its complete three-year outcome-blind data build. It
does not authorize model research, a profitability claim, shadow operation, or
production use.

## Why V39 Exists

V38 tested intraday index-futures and ETF structure, but the repository's
Tushare account lacks both required historical minute permissions. V39 keeps
the independent derivatives-risk hypothesis while replacing inaccessible
intraday data with information fully known after T-1 closes.

The V39 data is therefore available before every T-day 14:20-14:50 decision.
No target-day derivative observation is allowed.

## Independently Audited Source Families

### T-1 index futures daily

Frozen instruments:

| Family | Continuous future | Underlying index |
| --- | --- | --- |
| CSI 300 | `IF.CFX` | `000300.SH` |
| SSE 50 | `IH.CFX` | `000016.SH` |
| CSI 500 | `IC.CFX` | `000905.SH` |
| CSI 1000 | `IM.CFX` | `000852.SH` |

The actual main contract is resolved by `fut_mapping` on every T-1 source
date. `fut_daily` supplies all CFFEX contracts and `index_daily` supplies the
underlying close.

Frozen features include:

- main-contract close and settlement return;
- open-interest change and volume/open-interest ratio;
- settlement basis versus the underlying index;
- next-contract term spread;
- main-contract share of family open interest;
- large-, mid-, and small-cap cross-family basis and positioning summaries.

### T-1 ETF options daily

Frozen underlyings:

- `510050.SH`;
- `510300.SH`.

`opt_basic` supplies immutable contract metadata, `opt_daily` supplies T-1
option settlement, volume, amount, and open interest, and `fund_daily`
supplies the underlying ETF close.

Frozen features include:

- put/call volume, amount, and open-interest ratios;
- nearest 7-45 calendar-day ATM straddle as a percentage of spot;
- ATM put/call premium ratio;
- option volume/open-interest ratio;
- top-five open-interest concentration;
- cross-underlying option-risk summaries.

## Immutable Probe Dates

- `20230825`;
- `20231229`;
- `20240315`;
- `20240927`;
- `20250115`;
- `20250723`;
- `20260115`;
- `20260723`.

For each date, the source date is resolved from the SSE trade calendar and
must be strictly earlier than the target date.

## Frozen Gates

Each source family is audited separately. A failed family cannot be hidden by
a passing family.

Common gates:

1. exactly eight unique target-date rows;
2. every source date is the immediately preceding SSE trading date;
3. no target-day observation is read;
4. no T+1, truth, exit, label, gross-return, net-return, or other outcome field
   is read or emitted;
5. every frozen feature for the family is finite on all eight dates;
6. at least 80% of the family's features have at least four distinct probe
   values.

Additional futures gates:

1. all 32 `fut_mapping` queries succeed;
2. every mapping is unique;
3. all required main and next contracts, index closes, settlement prices,
   volumes, and open-interest fields exist.

Additional options gates:

1. `opt_basic`, all eight SSE `opt_daily` queries, and all 16 ETF daily
   queries succeed;
2. both underlyings have active calls and puts;
3. every date has a paired nearest-expiry ATM call and put with 7-45 days to
   maturity.

Missing permission, empty data, ambiguous mapping, or insufficient variation
fails that source family. No gate may be relaxed after seeing the result.

## Decision Rule

If neither family passes, V39 closes before outcomes are read.

If one or both families pass, only the passing family or families may enter a
complete three-year point-in-time build. After that data build is frozen, one
nested walk-forward protocol may be preregistered and joined to the existing
T+1 net-return outcomes.

Historical success can authorize only an unchanged 150-future-trading-day
shadow run.

# WP V38 Index-Futures Regime Data Feasibility Protocol

## Status

This outcome-blind protocol is frozen before any V38 candidate return is read.
A pass authorizes only a complete three-year point-in-time data build. It does
not authorize model research, shadow operation, public candidates, production,
or a profitability claim.

## Why This Direction Is Independent

V23 through V37 already tested stock OHLCV paths, opening auction, money flow,
positioning, fine-industry peers, public events, limit-up ecology, full-session
stock paths, regime gating, and alternative entry confirmation. Most later
models still ranked a legacy stock opportunity set and did not add a genuinely
independent forward-looking market-risk input.

V38 tests a different mechanism: whether professional index-hedging activity,
open-interest changes, futures-versus-tracking-ETF return divergence, and
large-cap versus small-cap tail rotation identify regimes in which a stock
signal is more likely to survive to T+1.

## Frozen Instruments

The following liquid, long-lived pairs are fixed before data access:

| Pair | Continuous future | Tracking ETF |
| --- | --- | --- |
| CSI 300 | `IF.CFX` | `510300.SH` |
| SSE 50 | `IH.CFX` | `510050.SH` |
| CSI 500 | `IC.CFX` | `510500.SH` |
| CSI 1000 | `IM.CFX` | `512100.SH` |

The actual futures month must be resolved by `fut_mapping` for every trade
date. A missing or ambiguous mapping fails the probe.

## Immutable Probe Dates And Slots

Probe dates:

- `20230825`;
- `20231229`;
- `20240315`;
- `20240927`;
- `20250115`;
- `20250723`;
- `20260115`;
- `20260723`.

Slots are `14:20`, `14:25`, `14:30`, `14:35`, `14:40`, `14:45`, and `14:50`.
The probe must emit exactly 56 unique date-slot rows.

## Historical And Live Symmetry

- futures mapping: Tushare `fut_mapping`;
- historical ETF source: `etf_mins`, frequency `1min`;
- historical futures source: `ft_mins`, frequency `1min`;
- live ETF source: `rt_etf_min_daily`, frequency `1MIN`;
- live futures source: `rt_fut_min_daily`, frequency `1MIN`;
- both historical and live frames normalize to the same timestamp, OHLC,
  volume, amount, and futures open-interest schema;
- every feature is truncated at or before its signal timestamp;
- mutating all post-signal rows must not change an earlier feature row.

The ETF and futures prices are on different scales. V38 therefore does not
claim to measure absolute cash-futures basis. Its hedge spread is explicitly
defined as futures return minus tracking-ETF return over the same causal
window.

## Frozen Outcome-Blind Features

For each of the four pairs:

- ETF and futures return since the continuous-auction open;
- ETF and futures trailing-20-minute return;
- matched-window futures-minus-ETF hedge spreads;
- futures open-interest change since open and over 20 minutes;
- ETF and futures trailing-20-minute amount shares;
- one-minute tracking error and ETF/futures return correlation.

Cross-pair summaries:

- CSI 1000 and CSI 500 rotation versus CSI 300;
- corresponding futures rotation;
- mean and dispersion of hedge spreads;
- mean and dispersion of open-interest changes.

## Frozen Probe Gates

Every gate must pass:

1. all 32 continuous-contract mapping queries succeed and are unique;
2. all 64 historical minute queries succeed;
3. all four pairs exist on all eight dates;
4. every date-slot pair has at least 98% expected ETF and futures minutes;
5. futures open interest is finite on at least 98% of causal bars;
6. all prices are finite and positive;
7. all 56 date-slot identities are present exactly once;
8. every latest timestamp is at or before its own slot;
9. at least 98% of rows have a finite feature vector;
10. at least 80% of frozen features have at least ten distinct probe values;
11. every probe date has all seven complete slot rows;
12. historical and live frames normalize to equivalent schemas in tests;
13. post-signal mutation cannot change earlier features;
14. no T+1, target, truth, exit, gross-return, or net-return field is read or
    emitted.

Any missing API permission is a data-direction failure, not a reason to relax
the contract.

## Decision Rule

A failure closes V38 before outcomes are read. A pass authorizes a full
three-year outcome-blind data build using the identical instruments, mappings,
features, and cutoffs.

Only after that build is frozen may one nested walk-forward model protocol be
registered and joined to T+1 outcomes. Historical success can authorize only
an unchanged 150-future-trading-day shadow run.

# WP V34 Full-Session Path Data Build Protocol

## Status

This outcome-blind full-data protocol is frozen after the V34 feasibility
probe passed and before any V34 candidate return is read. Passing this build
authorizes only a separately preregistered nested out-of-sample model
evaluation. It does not authorize production, shadow operation, public
candidates, or a profitability claim.

## Frozen Sources

- immutable V24 top-five candidate run: `30635569735`;
- immutable V34 feasibility probe run: `30676761165`;
- historical minute API: Tushare `stk_mins`, frequency `1min`;
- live counterpart reserved for a later integration gate: Tushare
  `rt_min_daily`, frequency `1MIN`.

The complete V24 candidate index is retained. The build may read only
candidate identity, fold, signal slot, and signal price. No candidate T+1
return, target, label, truth, or exit field may be read.

## Query And Time Contract

Queries are grouped by immutable candidate stock and calendar month. Each
stock-month request covers 09:25 through 14:50, but feature construction keeps
only continuous-auction observations and truncates every row at its own
14:20-14:50 signal timestamp.

The signal price is used only to audit that the last available one-minute
close at the signal matches the immutable candidate source within ten basis
points. It is not an outcome or a model feature.

## Frozen Feature Contract

The 35 V34 features and all formulas are exactly those in the passed probe:

- full-session and sub-session returns, ranges, volatility, and path
  efficiency;
- drawdown, rebound, reversal, VWAP control, and close position;
- amount distribution and recent acceleration;
- bar-direction signed-amount proxies and morning/afternoon agreement;
- price-amount correlation, Amihud proxy, path-extreme timing,
  autocorrelation, and recent volatility regime.

No feature may be added, removed, redefined, or selected using T+1 outcomes
during this build.

## Full-Data Gates

The complete dataset passes only if:

1. the V24 source manifest, row count, identities, and digest are valid;
2. the V34 probe manifest and feature digest are valid;
3. every required stock-month query succeeds;
4. feature identities exactly equal all immutable V24 candidate identities;
5. at least 98% of rows have at least 98% of expected causal minutes;
6. at least 98% of rows match signal price within ten basis points;
7. at least 98% of rows contain a finite 35-feature vector;
8. every feature timestamp is at or before its own signal;
9. every candidate date has at least 95% complete rows;
10. at least 90% of frozen features have meaningful cross-sample variation;
11. no forbidden outcome field is emitted;
12. all eight probe dates reproduce candidate identities, quality fields, and
    every numeric feature within absolute tolerance `1e-12`.

Any failure stops V34 model research until the data contract is diagnosed.
Rows may not be dropped because their later returns are inconvenient.

## Next Gate

After a pass, a model-and-decision protocol must be frozen before outcomes are
joined. That protocol must specify:

- nested date-based train, validation, and untouched test folds;
- target as positive net return under the fixed T+1 close exit after costs;
- candidate publication thresholds and zero-candidate behavior;
- minimum frequency, uncertainty, stress, and clustered-confidence gates;
- one final confirmatory test with no post-test tuning.

Only a passed historical protocol may begin an unchanged 150-future-trading-day
shadow run.

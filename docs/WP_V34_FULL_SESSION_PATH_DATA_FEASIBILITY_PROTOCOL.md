# WP V34 Full-Session Path Data Feasibility Protocol

## Status

This outcome-blind protocol is frozen before any V34 candidate return is read.
A pass authorizes only a complete three-year point-in-time data build. It does
not authorize model research, shadow operation, public candidates, production,
or a profitability claim.

## Independent Hypothesis

V22 established that recombining the existing tail five-minute and lagged
daily features is not a credible continuation. V23 and V24 added one-minute
microstructure only from 13:55 through the signal, the completed opening
auction, and previous-day money flow; that family also failed.

V34 tests a distinct mechanism: whether the complete causal path from the
continuous-auction open through each 14:20-14:50 signal contains information
about demand persistence, intraday exhaustion, VWAP control, session timing,
and flow-regime agreement that was absent from the short tail window.

## Immutable Candidate Source

- V24 outcome-blind top-five candidate data run: `30635569735`;
- sample dates:
  - `20230825`;
  - `20231229`;
  - `20240315`;
  - `20240927`;
  - `20250115`;
  - `20250723`;
  - `20260115`;
  - `20260723`;
- all V24 candidates on those dates are retained;
- only identity, fold, signal price, and signal slot are read;
- no T+1, return, label, target, truth, or exit column is read.

## Historical And Live Symmetry

- historical source: Tushare `stk_mins`, frequency `1min`;
- live source: Tushare `rt_min_daily`, frequency `1MIN`;
- both normalize to stock code, timestamp, OHLC, volume, and amount;
- every feature row is cut at or before its own signal timestamp;
- lunch-break and post-signal rows cannot affect an earlier feature row.

V34 may use only continuous-auction bars from 09:30 through 11:30 and 13:00
through the current signal. The exact live integration is a later gate; this
probe establishes historical coverage and a shared normalized schema.

## Frozen Feature Family

The probe computes only causal full-session path summaries:

- opening-30-minute, morning, afternoon, post-14:00, and full-session returns;
- lunch gap, morning and afternoon range, realized and downside volatility;
- directional efficiency, drawdown, rebound, reversal, VWAP gap, close
  position, and shares of minutes above VWAP;
- opening, morning, and last-30-minute amount shares plus recent amount
  acceleration;
- price-direction signed-amount proxies for the morning, afternoon, post-14:00,
  and complete path;
- morning/afternoon flow agreement, price-amount correlation, Amihud proxy,
  high/low timing, time since extremes, zero-amount share, return
  autocorrelation, and recent-versus-prior volatility.

Signed amount is explicitly a bar-direction proxy, not exchange aggressor-side
order-flow truth.

## Frozen Probe Gates

The data family passes only if every gate passes:

1. all stock-date minute queries succeed;
2. all eight dates are represented;
3. candidate identities exactly match the immutable source;
4. at least 98% of rows contain at least 98% of expected session minutes;
5. at least 98% of rows match the immutable signal price within 10 basis
   points;
6. at least 98% of rows have finite feature vectors;
7. every latest timestamp is at or before its signal;
8. at least 90% of frozen features have at least ten distinct probe values;
9. each probe date has at least 95% complete rows;
10. no forbidden outcome field is read or emitted;
11. mutating every post-signal bar leaves earlier features byte-equivalent in
    contract tests;
12. historical and live minute responses normalize to the same schema in
    contract tests.

## Decision Rule

Any failed gate closes V34 before outcome research. A pass authorizes only a
full three-year outcome-blind data build with the identical source, cutoffs,
features, and coverage rules.

A separate model protocol must then be frozen before joining T+1 outcomes.
Any historical success can authorize only an unchanged 150-future-trading-day
shadow run.

# WP V23 Point-in-Time Data Protocol

## Purpose

V23 introduces genuinely new causal information after V20, V21, and V22
failed on the existing five-minute and lagged-daily feature base. This stage
builds data only. It does not inspect profit outcomes, tune a model, select a
threshold, or authorize live use.

## Fixed Source

- Immutable source run: V19 full-universe V9 walk-forward shards.
- Opportunity set: the fixed V19 broad-recall construction with
  `top_per_source=48` and `exploration_per_slot=24`.
- Source leader: one source-eligible V9 leader per trade date and signal slot,
  ranked by the already-frozen V9 selection score.
- Signal slots: 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, and 14:50.
- The source projection excludes all profit, target, T+1, truth, and exit
  columns before any V23 data request is constructed.

The set of required stock-month and stock-date requests is therefore fixed
without consulting V23 outcomes.

## New Causal Data

### Historical One-Minute Bars

For each fixed source leader, V23 reads one-minute bars from 13:55 through the
signal slot. Bars after the signal timestamp are forbidden for that row.

The feature family includes short-window returns, realized and downside
volatility, return skew and autocorrelation, directional efficiency, drawdown
and rebound, minute direction streaks, signed-amount imbalance proxy, Amihud
impact proxy, amount concentration and acceleration, VWAP gap, close position,
wick pressure, and local break counts.

The signed-amount value is a price-direction proxy. It is not represented as
exchange order-flow truth.

### Same-Day Opening Auction

Only the completed opening auction from the same trade date is used. Features
include the opening gap from the causally reconstructed previous close,
auction return and range, close position, VWAP gap, amount, and volume.

### Previous-Day L2 Money Flow

For every T-day signal, the money-flow source date must equal the immediately
preceding open A-share trading date. Same-day and future money flow are
forbidden. Features include normalized total, large, medium, and small flow,
institution-versus-retail spread, and gross flow scale.

## Coverage Contract

- Per-row one-minute coverage must be at least 90%.
- Dataset-level minute, auction, previous-day money-flow, and complete-row
  coverage must each be at least 98%.
- Identity keys must be unique.
- The latest minute timestamp must not exceed the signal timestamp.
- The money-flow source date must be strictly earlier than the signal date.
- Missing rows are determined before outcomes are joined.
- Missing data may mark a row unavailable but can never be filtered based on
  its return.

If any coverage gate fails, V23 model research is not authorized.

## Immutable Evidence

The build emits:

- an outcome-blind source leader index;
- raw, filtered one-minute partitions by month;
- raw, filtered opening-auction partitions by year;
- raw, filtered previous-day money-flow partitions by year;
- one point-in-time feature row per source leader;
- source hashes, partition hashes, query failures, and coverage audit in a
  manifest.

Only after this manifest passes may a separately preregistered V23 nested
out-of-sample model protocol be executed.

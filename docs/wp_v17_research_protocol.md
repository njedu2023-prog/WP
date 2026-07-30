# WP V17 Research Protocol

## Objective

V17 has one objective:

> Maximize the probability that every executable candidate first published
> between 14:20 and 14:50 earns a positive net return when sold at the
> immutable T+1 close, after the established round-trip costs.

The system may publish multiple candidates. The human user decides whether and
which candidate to buy. A `NO_SIGNAL` day is valid and must never be converted
into a trade by lowering thresholds.

## Frozen Contract

- Candidate clock: T day, 14:20 through 14:50 inclusive.
- Entry record: the first qualifying signal time and contemporaneous price.
- Exit: T+1 close auction.
- Objective label: net return after the established all-in round-trip cost.
- Duplicate rule: one immutable first signal per trade date and stock.
- Post-close rule: no buyable candidate remains visible after 15:00.
- Human fills and model candidates remain separate records.

V15 remains unchanged. V16's six-specialist intersection is a failed research
branch and is not reused by V17.

## Data And Leakage Control

V17 consumes only the immutable V9 out-of-sample score frontier and the V11
executable T+1 truth contract. Every outer test date is later than all model
training, calibration, policy design, and policy confirmation dates.

The three-year source interval is reported in full. Model metrics are reported
only for outer folds that have enough strictly earlier data. Warm-up dates are
not mislabeled as model test results.

## Selector

The selector is deliberately low-dimensional:

1. A regularized tree-linear blend predicts positive net return.
2. A robust regression predicts expected net return.
3. A quantile regression predicts the 25th-percentile net return.
4. A separate calibration interval fixes probability and return calibration
   before each outer test fold.
5. All rows receive a cross-sectional score rank using only information
   available at the signal time.

No industry narrative, stock identity, calendar-year indicator, or result from
V16 is allowed into the feature set.

## Policy Family

Only 24 policies are tested. They vary four declared controls:

- conservative positive-return probability;
- minimum expected net return;
- cross-sectional score rank;
- daily candidate cap of two or three.

Severe-loss probability, executable round-trip fill, model disagreement, and
the return lower-quantile constraints are fixed across the family. The design
period applies Benjamini-Hochberg false-discovery control at `q <= 0.10`. The
single design champion must then pass an untouched confirmation period before
it can be applied to the next outer fold.

## Historical Gates

Historical readiness requires all of the following:

- at least 250 nested out-of-sample candidates;
- at least 50 candidate days;
- win rate at least 55%;
- Wilson and trade-day clustered win-rate lower bounds at least 52%;
- mean net return at least 0.20%;
- trade-day clustered mean-return lower bound above zero;
- Profit Factor at least 1.20;
- nonnegative mean after an additional real 50 bp per trade;
- 10th return percentile no worse than -3%;
- complete temporal-integrity audit.

Failure of any gate means `NOT_READY_FOR_SHADOW`.

## Promotion

Passing historical gates only creates a frozen shadow candidate. It does not
authorize production.

The exact model, feature list, policy, costs, and exit contract must then remain
unchanged for at least 150 future A-share trading days. All qualifying signals
and `NO_SIGNAL` days are logged. Production review starts only after that
future period and must use the same net-return contract.

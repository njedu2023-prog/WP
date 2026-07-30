# WP V14 Exit-Failure Risk Protocol

## Research question

V11 and V13 showed that changing the fixed T+1 sell time did not turn the
immutable V10 candidates profitable. Two of the 28 V10 candidates incurred the
predeclared failed-exit penalty. V14 tests a narrower causal question:

> Can information available at the T-day signal identify candidates that are
> at high risk of being unable to execute the fixed T+1 close-auction exit?

V14 does not alter the entry model, entry price, exit price, costs, or truth
labels.

## Frozen sources

- V9 causal point-in-time feature panel: run `30466227350`
- V10 immutable selected candidates: run `30516136872`
- V11 immutable candidate frontier and close-exit truth: run `30545808015`
- The V11 frontier SHA-256 is checked before any model is trained.

## Causal model

- Input: the registered T-day `FEATURE_COLUMNS` available at each signal.
- Target: T+1 close-auction exit non-fill under the immutable contract.
- Training universe: prior entry-fillable rows from the fixed V11 frontier.
- Model: 70% histogram gradient boosting and 30% regularized logistic
  probability, followed by isotonic calibration on an independent period.
- Class weighting is fixed from the training failure prevalence.
- Features with no information in a training window are excluded and the
  resulting feature subset is frozen for that fold.

## Walk-forward boundaries

For every outer V10 fold that contains at least one selected candidate:

1. Use 126 prior trading days for model fitting.
2. Purge two trading days.
3. Use 42 prior trading days for calibration.
4. Purge two trading days before the frozen outer-fold test.
5. Never use the outer-fold truth to train, calibrate, or choose a threshold.

## Fixed overlays

The following overlays are declared before reading V14 test outcomes:

- absolute failure probability at or below 0.5%, 1%, 2%, 3%, or 5%;
- within-date and within-slot safest 10%, 20%, or 50%;
- the existing fixed fill-probability gate;
- the 2% failure-risk gate combined with the existing fill gate.

The baseline is the same causally scored V10 subset. Results also include the
full 28-candidate V10 baseline so any unscored coverage is visible.

## Evidence standard

A historical direction must retain at least 10 candidates over at least five
trading days and have:

- positive mean net return;
- profit factor above 1;
- positive total return under 50bp all-in cost stress.

This is not sufficient for production. Statistical confirmation still requires
at least 250 out-of-sample candidates, positive day-clustered mean lower bound,
day-clustered win-rate lower bound of at least 52%, profit factor of at least
1.20, exit fill rate of at least 98%, and a new frozen 150-trading-day shadow
run.

V14 cannot authorize production regardless of its historical result.

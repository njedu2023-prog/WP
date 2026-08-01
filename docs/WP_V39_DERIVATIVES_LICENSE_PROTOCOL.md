# WP V39 T-1 Derivatives Market-License Protocol

## Status

This protocol is frozen after the outcome-blind V39 data build passed and
before any V39 candidate outcome is evaluated.

V39 is research-only. Passing every historical gate can authorize only an
immutable 150-A-share-trading-day future shadow run. It cannot authorize
production.

## Objective

Maximize the probability that an executable candidate first published from
14:20 through 14:50 on T earns a positive net return under the fixed T+1 close
exit after the established entry, fill, failure, and round-trip cost contract.

The system may publish zero candidates. It must not lower a threshold to force
a list. Every published candidate is evaluated, regardless of whether the
human user buys it.

## New Information Family

V22 and V35 rejected market-state gates built from the existing stock and
same-day path families. V39 introduces only data that was independently proven
available after T-1 close:

- CFFEX IF, IH, IC, and IM main-contract returns, basis, term structure,
  volume/open-interest, open-interest change, and main-contract concentration;
- SSE 50ETF and 300ETF option put/call volume, amount and open-interest ratios,
  near-ATM straddle premium, put/call premium ratio, volume/open-interest, and
  open-interest concentration.

The outcome-blind three-year build is fixed to run `30687183695`, artifact
`8814417777`, digest
`sha256:1fad354f4694b5270e7e0f7a229a66538554aab31bda76f46fcd572d229375ed`.
It contains exactly 725 target dates. Both data families have 100% complete and
finite coverage, all 51 features vary, all previous-trading-day mappings are
exact, and there were zero query failures.

## Immutable Source And Execution Contract

- Source: immutable V9 nested out-of-sample shards from run `30600193544`.
- Signal slots: 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, and 14:50.
- Source identity: the existing outcome-blind V22 slot leader after the frozen
  execution, fill, source-risk, disagreement, and freshness filters.
- Entry: the established executable signal-price contract.
- Exit: the established fixed T+1 close contract.
- Return: net of all established round-trip costs and failed-exit treatment.
- Stress: subtract a further 50bp from every candidate.
- First signal: once a stock qualifies on a date, its earliest qualifying
  signal time and price are immutable.
- Daily maximum: three distinct candidates.
- Human action: the user may buy none or one; model statistics always include
  every published candidate and remain separate from actual trades.

## Frozen Feature Contract

The license model sees:

- all 34 authorized T-1 futures features;
- all 17 authorized T-1 ETF-option features;
- the 17 preregistered V22 opportunity-set aggregates available at the signal
  slot;
- signal-slot minute.

The license model does not see stock code, selected-stock identity,
selected-stock raw features, T+1 values, return labels, future prices, close
truth, or post-signal fields.

At least 45 derivative features and ten opportunity-set aggregates must remain
active in every fitted fold. Features with insufficient finite variation in a
prior fold may be dropped without looking at test outcomes.

## Nested Out-Of-Sample Design

For each immutable V9 outer fold:

- train on the previous 252 A-share trading days;
- purge two trading days;
- calibrate on the next 42 prior trading days;
- purge two more trading days before the untouched outer test;
- give every stock-day equal total weight before fixed recency weighting;
- fit one shallow, strongly regularized histogram tree and one strongly
  regularized linear model for each of positive return, return above +0.50%,
  severe loss at or below -2.00%, and clipped expected net return;
- blend tree and linear estimates 70/30;
- calibrate all outputs only on the prior calibration segment.

No outer-test outcome may affect its feature set, model, calibrator, threshold,
candidate identity, signal time, or signal price.

## Frozen Release Policy

There is exactly one policy:

- target candidate-day rate: 25%, used only to take an unlabeled daily-maximum
  score quantile from the prior calibration dates;
- maximum candidates per day: three;
- probability-model spread: at most 0.35;
- expected-return-model spread: at most 4.00 percentage points;
- legal signal slots only;
- first qualifying appearance of each stock is retained;
- later appearances cannot replace the first signal;
- `NO_SIGNAL` is valid.

The fixed economic score is:

```text
expected net return lower bound
+ 1.00 * (positive-return probability lower bound - 0.50)
+ 0.60 * (margin probability lower bound - 0.30)
- 1.10 * severe-loss probability upper bound
```

No score weight, feature, policy, threshold rule, time slot, cost, subgroup, or
calendar subset may change after V39 outcomes become visible.

## Historical Acceptance Gates

Every gate must pass:

- at least 100 nested out-of-sample candidates;
- at least 70 candidate days;
- candidate-day rate from 15% through 35%;
- win rate at least 55%;
- Wilson win-rate lower bound at least 50%;
- trade-date-clustered win-rate lower bound at least 48%;
- +0.50% margin-hit rate at least 40%;
- -2.00% tail-loss rate at most 15%;
- mean net return at least +0.25%;
- trade-date-clustered 95% lower bound for mean net return above zero;
- Profit Factor at least 1.25;
- mean net return remains nonnegative after another 50bp cost;
- 10th-percentile return at least -3.00%;
- candidates in at least three calendar years;
- at least 20 candidates in every active calendar year;
- at least three positive calendar years;
- worst active calendar-year mean at least -0.10%;
- complete temporal, source, data, feature, signal, and outcome integrity.

## Decision Rule

- If any historical gate fails, V39 is permanently rejected. Its outputs,
  thresholds, features, years, and subgroups cannot be repaired after seeing
  the result.
- If all historical gates pass, freeze the resulting bundle and run it without
  modification for at least 150 future A-share trading days, with at least 45
  candidates on at least 30 candidate days.
- Production remains prohibited until the untouched shadow promotion contract
  also passes.

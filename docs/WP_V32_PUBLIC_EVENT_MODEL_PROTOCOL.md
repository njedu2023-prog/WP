# WP V32 Public-Event Model Protocol

## Frozen Question

The immutable V32 outcome-blind data source is GitHub Actions run
`30665412715`. This protocol becomes valid only if that run completes
successfully and its manifest sets `v32_model_research_authorized=true`.

The data contract passed without reading candidate outcomes:

- workflow job: `91271145335`;
- immutable artifact: `8808328109`;
- artifact name: `wp-v32-public-event-data-30665412715`;
- artifact digest:
  `sha256:93ca2d037905e681cdf96529f65599c367b9dc224c41438d4480bed7db671eed`;
- manifest authorization: `v32_model_research_authorized=true`.

V32 asks one question: do causal, A-share-normalized public events disclosed in
the five immediately prior trading days identify a practical-frequency set of
executable 14:20-14:50 candidates with positive net economics under the fixed
T+1 close exit?

There is one model family, one release policy, and one acceptance rule. Any
failed gate rejects V32. Features, labels, weights, probability gates, score
weights, candidate caps, slots, years, subgroups, and exit rules may not be
changed after the first nested out-of-sample result is read.

## Execution And Candidate Contract

- immutable V9/V24 top-five source candidates per date and slot;
- signal slots: 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, 14:50;
- established executable next-five-minute-close entry and slippage contract;
- established fixed T+1 close-auction exit;
- established round-trip costs and failed-exit treatment;
- at most one score leader per date and signal slot;
- at most three immutable first signals per trade date;
- multiple candidates may be published and the human chooses;
- `NO_SIGNAL` is valid and no gate may be relaxed to force a list.

## Information Boundary

Only these preregistered V32 event features may drive alpha:

- `forecast`;
- `repurchase`;
- `share_float`;
- `block_trade`.

Each source contributes prior-five-trading-day count, active flag, latest
trading-day age, and the numeric details frozen in
`WP_V32_FULL_EVENT_DATA_BUILD_PROTOCOL.md`.

Existing V9/V24 values are restricted to execution and data-quality controls:

- round-trip fill lower bound at least 0.95;
- source severe-loss probability at most 0.45;
- market-data age no more than 420 seconds when available;
- complete point-in-time and V32 event records.

Old model scores, post-signal prices, same-day close, T+1 values, outcomes,
truth, text categories, and artificial user choices are not V32 alpha
features. All model features must be selected from the frozen V32 event column
set using training and calibration data only. Profit outcomes were not read
during the data build.

## Frozen Nested Model

- prior training window: 252 A-share trading days;
- purge: 2 trading days;
- prior calibration window: 42 trading days;
- final purge before each outer test fold: 2 trading days;
- minimum training rows: 4,000;
- minimum calibration rows: 800;
- minimum pairwise training rows: 12,000;
- minimum pairwise calibration rows: 2,000;
- outer tests: immutable V9 folds;
- each repeated stock-day has equal total weight;
- each date-slot pairwise group has equal total weight.

Four regularized logistic heads use exactly the active V32 feature columns:

1. probability net return is positive;
2. probability net return exceeds +0.50%;
3. probability net return is at or below -2.00%;
4. probability stock A beats stock B in the same date and signal slot.

All use median imputation, standardization, logistic `C=0.04`, balanced class
weights for the pointwise heads, and prior-window isotonic calibration. The
pairwise sample contains both A-B and B-A differences. Rows without an event
remain in model training as the causal baseline, but only event-active rows may
qualify for release.

## Frozen Score And Release Policy

Hard gates:

```text
P(net positive) >= 0.50
P(net > 0.50%) >= 0.35
P(net <= -2.00%) <= 0.30
same-slot pairwise score >= 0.50
```

Score:

```text
1.00 * (P(net positive) - 0.50)
+ 0.75 * (P(net > 0.50%) - 0.35)
+ 1.00 * (same-slot pairwise score - 0.50)
- 1.00 * P(net <= -2.00%)
```

After all fixed gates, the highest score in each date-slot proceeds. The score
threshold is calibrated only from the prior 42-day window to target candidate
days on 25% of calendar dates; calibration does not optimize historical
returns. The first qualifying signal for a stock-day is immutable, and the
first three qualifying stock-days are retained.

## Historical Acceptance Gates

All V24 economic gates must pass:

- at least 120 candidates over at least 80 candidate days;
- candidate-day rate between 12% and 35%;
- win rate at least 55%;
- Wilson lower bound at least 50%;
- trade-date-clustered win-rate lower bound at least 48%;
- +0.50% hit rate at least 40%;
- -2.00% tail-loss rate at most 15%;
- mean net return at least +0.20%;
- clustered mean-return lower bound above zero;
- profit factor at least 1.20;
- nonnegative mean after an additional 50bp cost;
- 10th percentile no worse than -3.00%;
- at least three active and three positive calendar years;
- at least 20 candidates in every active year;
- worst active-year mean at least -0.10%;
- all temporal, identity, source, feature, and outcome checks pass.

The event-active same-slot mechanism must also pass:

- at least 1,000 evaluable date-slot groups;
- mean same-slot Spearman IC at least +0.05;
- mean highest-minus-lowest return spread at least +0.20%;
- trade-date-clustered 95% lower bound of the spread above zero;
- positive spread in at least three calendar years.

## Decision Rule

Any failed gate rejects V32. A complete historical pass only freezes a shadow
candidate; it does not authorize production. The unchanged bundle must then
complete at least 150 untouched future A-share trading days, with at least 60
candidates over at least 40 candidate days, before any production decision.

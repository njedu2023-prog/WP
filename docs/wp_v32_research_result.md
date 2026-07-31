# WP V32 Public-Event Research Result

## Decision

V32 is rejected. It is not authorized for the required 150-trading-day future
shadow run or for production.

The four causal public-event families had adequate historical coverage, but
the frozen nested model released only two candidates in 725 evaluation days.
The event-active same-slot ranking mechanism was also below every
preregistered information and economic gate. This is a valid negative model
result, not a data, identity, outcome, or workflow failure.

## Immutable Evidence

- Preregistered model and trigger commit:
  `6225bef301519f18e86273c909ba729e95388bd3`
- Immutable V9 source run: `30600193544`
- Immutable V24 point-in-time data run: `30635569735`
- Immutable V32 public-event data run: `30665412715`
- V32 research run: `30670186694`
- V32 research job: `91286059433`
- Research artifact: `8808560183`
- Research artifact name: `wp-v32-public-event-30670186694`
- Research artifact digest:
  `sha256:5ad7d09e02e485823d8d3ef5d513502e687aa963eebbcff21b8c62b3424b9aae`
- Research artifact size: `12,608,974` bytes
- Evaluation trading days: `725`
- Model-covered trading days: `725`
- Outcome-blind source rows: `31,955`
- Joined event-feature rows: `31,955`
- Verified selected outcomes: `2 / 2`
- Missing selected outcomes: `0`
- Inconsistent selected outcomes: `0`

## Nested Out-Of-Sample Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Candidates | at least 120 | 2 | no |
| Candidate days | at least 80 | 2 | no |
| Candidate-day rate | 12%-35% | 0.2759% | no |
| Win rate | at least 55% | 50.0000% | no |
| Wilson win lower bound | at least 50% | 9.4531% | no |
| Clustered win lower bound | at least 48% | 50.0000% | yes |
| Return above +0.50% | at least 40% | 50.0000% | yes |
| Return at or below -2.00% | at most 15% | 0.0000% | yes |
| Mean net return | at least +0.20% | +0.0034% | no |
| Clustered mean lower bound | above 0% | +0.0034% | yes |
| Profit factor | at least 1.20 | 1.0064 | no |
| Additional 50bp stress mean | at least 0% | -0.4966% | no |
| Return 10th percentile | at least -3.00% | -0.8603% | yes |

The two selected candidates were one `+1.0831%` result in 2023 and one
`-1.0762%` result in 2026. No candidate was released in 2024 or 2025. The
near-zero arithmetic mean and the passing two-row confidence fields have no
statistical meaning and cannot override the minimum-sample gates.

## Same-Slot Ranking Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Evaluable event-active date-slot groups | at least 1,000 | 945 | no |
| Mean same-slot Spearman IC | at least +0.05 | +0.0423 | no |
| Highest-minus-lowest return | at least +0.20% | -0.0204% | no |
| Clustered spread lower bound | above 0% | -0.1591% | no |
| Positive spread years | at least 3 | 2 | no |

Yearly highest-minus-lowest spreads were:

| Year | Groups | Mean rank IC | Same-slot spread |
| --- | ---: | ---: | ---: |
| 2023 | 111 | +0.0896 | +0.5414% |
| 2024 | 373 | +0.0865 | -0.0946% |
| 2025 | 314 | +0.0191 | +0.0533% |
| 2026 | 147 | -0.0558 | -0.4140% |

The recent deterioration and negative pooled top-minus-bottom spread show that
the event model does not reliably identify the better stock available at the
same live signal time.

## Failed Frozen Gates

V32 failed 17 gates:

1. minimum candidates;
2. minimum candidate days;
3. practical candidate-day rate;
4. minimum win rate;
5. minimum Wilson lower bound;
6. minimum mean net return;
7. minimum profit factor;
8. nonnegative additional-50-bp stress mean;
9. minimum three active calendar years;
10. minimum candidates in each active year;
11. minimum three positive calendar years;
12. worst-calendar-year mean;
13. minimum evaluable same-slot groups;
14. minimum same-slot rank IC;
15. minimum same-slot top-minus-bottom spread;
16. positive clustered same-slot spread lower bound;
17. positive same-slot spread in at least three years.

Any one failed gate rejects the model.

## What V32 Established

- Causal forecast, repurchase, share-float, and block-trade events can be built
  point-in-time with useful coverage and exact candidate identities.
- Adequate data coverage did not translate into predictive or economic value.
- The fixed probability gates were too selective, but lowering them is not a
  valid remedy because the independent same-slot ranking diagnostics were also
  weak and economically negative.
- Public events from the prior five trading days are not a credible standalone
  stock-selection mechanism for this fixed tail-entry and T+1-close contract.

## Closed Decisions

- Do not deploy or shadow V32.
- Do not relax V32 probability or release gates.
- Do not tune V32 features, windows, thresholds, score weights, candidate caps,
  years, slots, or subgroups after reading this result.
- Do not combine V32 with previously rejected models and describe the
  combination as confirmatory evidence.
- Preserve the V32 event dataset for audit and for explicitly preregistered
  orthogonal hypotheses only.

Any continuation must preregister a genuinely different causal mechanism and
must not use the V32 release result to mine a favorable threshold or subgroup.

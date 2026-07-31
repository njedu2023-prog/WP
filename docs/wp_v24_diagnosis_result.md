# WP V24 Failure-Attribution Result

## Decision

V24 remains rejected. The existing V9-V24 feature family does not contain a
stable, economically material same-day same-slot stock-ranking signal. No
successor policy, shadow model, or production model is authorized by this
diagnosis.

## Immutable Evidence

- Source V24 research run: `30643568519`
- Diagnosis workflow run: `30646607301`
- Diagnosis job: `91209422530`
- Source commit: `0c10f437eb325c03a2571ff4389a3efe70f859e2`
- Artifact: `8799818417`
- Artifact name: `wp-v24-failure-attribution-30646607301`
- Artifact digest:
  `sha256:e08401a07acb724333902ef60af7e6ce2a322b4bc1e55e5c563fb8e139e68d54`
- Scored rows: `25,375`
- Scored trading days: `725`
- Same-day same-slot cross-sections: approximately `5,000`
- Contract tests: `8 passed`

## Global Discrimination Is Not Stock Selection

The probability heads retained moderate global AUC, but the result disappears
or reverses when stocks are compared only with other stocks available at the
same decision time.

| Output | Global metric | Same-slot daily IC | Best-minus-worst return |
|---|---:|---:|---:|
| Positive probability | AUC `0.6631` | `-0.0548` | `-0.1127%` |
| Margin probability | AUC `0.6648` | `-0.0492` | `-0.1203%` |
| Severe-loss safety | AUC `0.7012` | `+0.0498` | `+0.0780%` |
| Expected net return | Spearman `0.0317` | `+0.0098` | `+0.0519%` |
| Economic score | Spearman `0.0090` | `-0.0147` | `-0.0364%` |

The AUC values therefore mostly describe favorable versus unfavorable market
dates. They do not show that the model can choose the better stock from the
same tradable opportunity set.

Lower predicted severe-loss risk had statistically detectable rank
information, but its average return spread was only `+0.0780%`, below the
preregistered `+0.20%` economic floor. Its yearly return spread was also
negative in 2023 and 2025. It may remain a risk control, but it is not an alpha
selector.

## Causal Feature Audit

- Causal features evaluated: `68`
- Stable exploratory features: `0`
- Minute microstructure best spread: `+0.0576%`
- Opening-auction best spread: `-0.1232%`
- Previous-day money-flow best spread: `-0.0841%`
- Source cross-section best spread: `+0.1008%`

No feature passed the joint requirements for sample size, multiple-testing
control, economic spread, and calendar stability.

## V24 Failure Is Broad

The released V24 candidates remain economically negative:

- Candidates: `488` on `225` trading days
- Win rate: `44.47%`
- Mean net return: `-0.1629%`
- Profit Factor: `0.8668`
- Additional 50bp stress mean: `-0.6629%`
- Return p10: `-3.7593%`

The worst ten candidates account for `17.58%` of total negative return points
and the worst twenty for `30.76%`. The loss is not repaired by removing a few
outliers.

## Next Admissible Research

Do not retune V24, invert a model output after seeing this result, or promote
the positive 14:40/fold-8 subgroups. The next study must:

1. add a genuinely independent point-in-time information family;
2. prove historical availability and causal timing before reading outcomes;
3. make same-day same-slot ranking an explicit acceptance gate;
4. retain the fixed executable entry, T+1 close exit, cost, and 150-future-day
   shadow contracts.


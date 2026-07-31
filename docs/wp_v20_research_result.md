# WP V20 Hierarchical Opportunity Research Result

## Decision

**Rejected. Do not promote to future shadow or production.**

V20 increased opportunity frequency but did not produce a reliable economic
edge under the preregistered fixed T+1 close contract. The model passed source
and temporal integrity, but failed eight profitability, uncertainty and
frequency gates.

## Immutable evidence

- Source V9/V19 run: `30600193544`
- Successful V20 research run: `30623421294`
- Research job: `91133008001`
- Evidence artifact: `8790504453`
- Artifact digest:
  `sha256:9cf95d5802f4307eb3ed9c833805f9816769a36e307b5ed22b94fe39b464af44`
- Evaluation window: `20230727` through `20260724`
- Evaluation trading days: `725`
- Model-covered days: `703`
- Causal leader rows: `19,173`
- Source integrity: passed
- Temporal integrity: passed

## Nested out-of-sample result

| Metric | Result | Required |
|---|---:|---:|
| Candidates | 302 | at least 50 |
| Candidate days | 192 | at least 30 |
| Candidate-day rate | 26.48% | 10%-25% |
| Win rate | 46.03% | at least 55% |
| Wilson win-rate lower bound | 40.49% | at least 48% |
| Day-clustered win-rate lower bound | 40.47% | at least 48% |
| Mean net return | +0.0291% | at least +0.20% |
| Day-clustered mean lower bound | -0.2485% | above 0% |
| Profit factor | 1.0334 | at least 1.20 |
| Additional 50bp stress mean | -0.4709% | at least 0% |
| Return 10th percentile | -2.4966% | at least -3.00% |

The small positive mean is not statistically or economically usable. It is
less than one tenth of the required mean, disappears under realistic cost
stress, and its clustered confidence interval includes material losses.

## Calendar stability

| Year | Candidates | Candidate days | Win rate | Mean net return | 50bp stress |
|---|---:|---:|---:|---:|---:|
| 2023 | 24 | 17 | 54.17% | +0.7692% | +0.2692% |
| 2024 | 69 | 43 | 37.68% | -0.1643% | -0.6643% |
| 2025 | 123 | 80 | 46.34% | -0.0766% | -0.5766% |
| 2026 | 86 | 52 | 50.00% | +0.1290% | -0.3710% |

V20 is therefore regime-dependent rather than stable. Two of four active
calendar years lost money before the additional stress charge.

## Interpretation

Separating an opportunity gate from stock ranking did not create enough alpha.
The fixed 18% calibration rate also drifted to a 26.48% realized candidate-day
rate as score distributions changed across folds. More stable frequency
normalization would fix that engineering symptom, but cannot repair the 46%
win rate or the negative 50bp stress result.

The next step is not to lower V20 thresholds or select a favorable subgroup
after seeing its outcomes. V20 candidate evidence may be used only for
exploratory loss attribution. Any V21 mechanism and acceptance contract must
be independent, frozen before execution, and remain research-only until at
least 150 genuinely future A-share trading days are observed.

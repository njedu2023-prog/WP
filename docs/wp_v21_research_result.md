# WP V21 Economic-Margin Research Result

## Immutable evidence

- Workflow run: `30624949710`
- Job: `91137919813`
- Source V9 shard run: `30600193544`
- Artifact: `8791116655`
- Artifact digest:
  `sha256:028d4d67947f1e83e252092e1d88bfb9be89dbcd9589f3faf2119ebca8f17144`
- Protocol: `wp_v21_economic_margin_1`
- Result: rejected

## Nested out-of-sample result

| Metric | Result |
|---|---:|
| Evaluation days | 725 |
| Model-covered days | 703 |
| Candidates | 231 |
| Candidate days | 158 |
| Candidate-day rate | 21.79% |
| Win rate | 48.92% |
| Wilson lower | 42.54% |
| Day-clustered win lower | 42.24% |
| Mean net return | +0.0377% |
| Median net return | -0.0384% |
| Profit Factor | 1.0315 |
| Additional 50bp stress | -0.4623% |
| Return p10 | -3.8172% |
| Margin hit rate, net > +0.50% | 35.93% |
| Tail loss rate, net <= -2.00% | 23.38% |

## Calendar stability

| Year | Candidates | Candidate days | Win rate | Mean net return | 50bp stress |
|---|---:|---:|---:|---:|---:|
| 2023 | 18 | 14 | 44.44% | -0.2987% | -0.7987% |
| 2024 | 52 | 39 | 53.85% | +0.0179% | -0.4821% |
| 2025 | 65 | 50 | 50.77% | +0.3367% | -0.1633% |
| 2026 | 96 | 55 | 45.83% | -0.0910% | -0.5910% |

## Failed gates

V21 failed 13 historical acceptance gates, including win rate and confidence
bounds, margin-hit rate, tail-loss rate, mean return, clustered mean lower
bound, Profit Factor, 50bp stress, return p10, and calendar stability.

Source integrity and temporal integrity passed. Therefore this is a genuine
negative result rather than a pipeline or leakage failure.

## Decision

V21 is not authorized for shadow operation or production. Changing the target
from positive net return to a +0.50% economic margin did not make the
stock-level second-stage ranking transferable. Further threshold tuning on V21
is prohibited.

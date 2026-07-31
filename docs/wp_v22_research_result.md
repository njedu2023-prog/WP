# WP V22 Market-License Research Result

## Immutable evidence

- Workflow run: `30626258207`
- Job: `91142137919`
- Source V9 shard run: `30600193544`
- Artifact: `8791553036`
- Artifact digest:
  `sha256:60d1ce814fe56c444409ef34080e54a4487e038bdcc60d83cc56ef0a7cc11a92`
- Protocol: `wp_v22_market_license_1`
- Result: rejected

## Nested out-of-sample result

| Metric | Result |
|---|---:|
| Evaluation days | 720 |
| Model-covered days | 698 |
| Market-slot leaders | 6,338 |
| Candidates | 138 |
| Candidate days | 112 |
| Candidate-day rate | 15.56% |
| Win rate | 44.20% |
| Wilson lower | 36.19% |
| Day-clustered win lower | 35.34% |
| Mean net return | -0.0429% |
| Median net return | -0.2957% |
| Profit Factor | 0.9502 |
| Additional 50bp stress | -0.5429% |
| Return p10 | -2.0825% |
| Margin hit rate, net > +0.50% | 36.23% |
| Tail loss rate, net <= -2.00% | 12.32% |

## Calendar stability

| Year | Candidates | Candidate days | Win rate | Mean net return | 50bp stress |
|---|---:|---:|---:|---:|---:|
| 2023 | 11 | 11 | 27.27% | -0.8902% | -1.3902% |
| 2024 | 27 | 22 | 51.85% | +0.2659% | -0.2341% |
| 2025 | 84 | 66 | 41.67% | -0.1527% | -0.6527% |
| 2026 | 16 | 13 | 56.25% | +0.5951% | +0.0951% |

## Failed gates

V22 failed 11 historical acceptance gates:

- win rate;
- Wilson and day-clustered win-rate lower bounds;
- economic-margin hit rate;
- mean net return and clustered mean lower bound;
- Profit Factor;
- additional 50bp cost stress;
- minimum three positive calendar years;
- worst-calendar-year stability.

Candidate frequency, tail-loss rate, return p10, source integrity, feature
integrity, and temporal integrity passed.

## Decision

V22 is not authorized for shadow operation or production. A separate
market-state permission layer did not rescue the unchanged V9 stock selector.

V20, V21, and V22 now provide converging evidence that repeatedly recombining
the same five-minute and lagged-daily feature base is not a credible next
step. Further model or threshold tuning on this historical evidence is
prohibited. The next research mechanism must introduce a genuinely new
point-in-time information family, such as reproducible intrabar microstructure
or order-flow evidence, while preserving the fixed execution and T+1 close
contracts.

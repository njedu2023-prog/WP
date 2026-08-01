# WP V39 T-1 Derivatives License Research Result

## Decision

V39 is permanently rejected.

The frozen T-1 CFFEX futures and SSE ETF-option state did not identify a
profitable, stable market regime for releasing the existing executable 14:20
through 14:50 stock opportunities. The result failed the historical promotion
contract and cannot authorize either a 150-trading-day shadow run or
production.

No V39 feature, score weight, threshold, subgroup, year, slot, or cost
assumption may be retuned after this result.

## Immutable Evidence

- frozen protocol commit:
  `4307758f032482fcf441db7d0608a189cea3c755`;
- audit-scope correction commit:
  `7f68679a5e46189accb138d0a66857d0327541b9`;
- final workflow run: `30688103727`;
- final job: `91337753583`;
- artifact: `wp-v39-derivatives-license-research-30688103727`;
- artifact id: `8814721415`;
- artifact digest:
  `sha256:d21add2872fbf2571039826ae23af022e3a9a28ed938e7edf18e6340265886b6`;
- artifact size: 2,925,813 bytes;
- immutable source run: `30600193544`;
- immutable derivatives-data run: `30687183695`.

## Nested Out-Of-Sample Result

- evaluation dates: 725;
- model-covered dates: 409;
- executable slot leaders: 6,338;
- released candidates: 203;
- candidate days: 101;
- candidate-day rate: 13.93%;
- profitable candidates: 88;
- win rate: 43.35%;
- Wilson win-rate lower bound: 36.72%;
- trade-date-clustered win-rate lower bound: 35.07%;
- mean net return: -0.0745%;
- median net return: -0.1585%;
- trade-date-clustered mean lower bound: -0.4080%;
- Profit Factor: 0.8959;
- return above +0.50%: 28.57%;
- loss at or below -2.00%: 9.85%;
- mean after another 50bp cost: -0.5745%;
- 10th-percentile return: -1.9741%.

## Calendar Stability

| Year | Candidates | Candidate days | Win rate | Mean net return |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 5 | 3 | 60.00% | +0.6392% |
| 2025 | 146 | 74 | 45.89% | +0.0636% |
| 2026 | 52 | 24 | 34.62% | -0.5308% |

The apparent 2024 result contains only five candidates. The strategy was
economically flat in 2025 and materially negative in 2026. This is not a
stable positive edge.

## Failed Frozen Gates

V39 failed:

- practical candidate-day frequency;
- raw, Wilson-lower, and clustered-lower win-rate gates;
- +0.50% margin-hit rate;
- minimum mean net return;
- positive clustered lower bound for mean return;
- minimum Profit Factor;
- additional 50bp cost stress;
- minimum candidates in every active year;
- three positive calendar years;
- worst-year return stability.

It passed candidate-count, candidate-day-count, severe-tail-loss,
10th-percentile, active-year-count, temporal-integrity, source-integrity, and
data-integrity gates.

## Audit Correction

The first run `30687830631` produced the same 203 candidates and exactly the
same economic metrics, but its summary incorrectly marked data integrity false
because it tested source rows outside the frozen 725-day derivatives window.

Commit `7f68679a5e46189accb138d0a66857d0327541b9` removed only that redundant
out-of-window boolean check. It did not change any feature, model, threshold,
candidate, signal time, signal price, return, or gate. The final rerun restored
data integrity to true while preserving every economic result.

## Research Conclusion

Daily T-1 futures basis, positioning, term structure, ETF-option put/call
state, and option-premium state are not a profitable release license for this
stock opportunity stream under the fixed execution and T+1-close contract.

V39 is closed. Its result may be used as negative evidence when choosing a
genuinely new information family, but it cannot be mined for a favorable
subset.

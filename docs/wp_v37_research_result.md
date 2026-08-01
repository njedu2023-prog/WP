# WP V37 Fast-Entry Research Result

## Verdict

**REJECTED. V37 is not authorized for production or the 150-day shadow gate.**

The preregistered hypothesis was that replacing the old delayed entry with a realistically published `t+2` signal and an exact `t+3` one-minute fill could recover enough edge to make the existing causal path ranker useful. The nested out-of-sample result rejects that hypothesis.

This result is final for protocol `wp_v37_fast_entry_contract_1`. It must not be tuned or rerun against the same historical outcome in response to this result.

## Immutable evidence

- Research commit: `2c541fa8919f7503874e6824f1cd7f0b27ea287b`
- GitHub Actions run: `30685179897`
- Evidence artifact: `wp-v37-fast-entry-research-30685179897`
- Artifact ID: `8813734922`
- Artifact ZIP SHA-256: `3396ad2b78664349d88ddd95fc8f691b3f3437b9a9a5d471ab04efc915bb953c`
- Frozen V9 causal source run: `30600193544`
- Frozen V24 point-in-time data run: `30635569735`
- Frozen V34 one-minute data run: `30677075531`

## Contract tested

- Signal bars: completed 5-minute bars from 14:20 through 14:45.
- Publication latency: signal published at `t+2` minutes.
- Entry: exact `t+3` one-minute close plus 10 bp adverse slippage.
- Exit: fixed T+1 adjusted close.
- Round-trip cost: 25 bp.
- Entry miss: cash return of zero, not a fabricated fill.
- T+1 exit miss after entry: fixed -10% conservative return.
- Nested folds: 252 training days, 2-day purge, 42 calibration days, 2-day purge, then out-of-sample evaluation.
- One preregistered policy only; no parameter sweep.

## Data and temporal integrity

- Evaluation days: **725**
- Model-covered days: **703**
- Source candidate rows: **31,955**
- Legal fast-entry rows: **27,390**
- Recomputed outcome rows: **27,390**
- Exact identity match: **passed**
- Duplicate identities: **0**
- Entry data completeness: **100%**
- Label availability: **100%**
- Exact entry timing audit: **passed**
- Selected outcome consistency: **passed**

The experiment is technically valid. The rejection is caused by economic and execution results, not missing data or temporal leakage.

## Nested out-of-sample result

| Metric | Result |
|---|---:|
| Selected candidates | 12 |
| Candidate days | 11 |
| Candidate-day frequency | 1.52% |
| Positive net-return candidates | 2 |
| Win rate | 16.67% |
| Wilson lower bound | 4.70% |
| Mean net return | -0.2702% |
| Median net return | 0.0000% |
| 10th percentile return | -1.5926% |
| Profit factor | 0.6701 |
| Mean net return after extra 50 bp stress | -0.7702% |
| Day-clustered mean lower bound | -1.2574% |
| Day-clustered win-rate lower bound | 0.00% |
| Entry fill rate | 58.33% (7/12) |
| T+1 exit fill rate after entry | 100% |

## Stability diagnostics

- 2023: no candidates.
- 2024: 5 candidates, 20.00% win rate, mean +0.3618%.
- 2025: 4 candidates, 25.00% win rate, mean -1.2629%.
- 2026: 3 candidates, 0 wins, mean 0.0000% because selected signals were not executable entries.
- Mean within-slot rank IC: **0.00624**.
- Top-minus-bottom return spread: **+0.03746%**, but its clustered lower bound is **-0.02508%**.
- The rank diagnostic turns negative in 2026.

There is no stable cross-year ranking edge and no practical signal frequency.

## Failed authorization gates

V37 failed 15 hard gates, including minimum sample size, candidate-day frequency, win rate, Wilson confidence, clustered stability, average net return, profit factor, 50 bp stress, per-year sample size, positive-year consistency, worst-year result, and minimum entry fill rate.

The only meaningful passes were data integrity, temporal integrity, tail-loss rate, 10th-percentile floor, and conditional T+1 exit availability. Those passes establish that the test was executable; they do not establish profitability.

## Research implication

The fast-entry timing change does not rescue the existing path ranker. Future research must not keep adjusting latency, thresholds, or policy quotas on this same label set. The next admissible direction requires a genuinely new information source or a separately held-out market regime, followed by a new preregistered test and ultimately at least 150 untouched future shadow trading days.

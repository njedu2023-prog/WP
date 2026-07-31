# WP V20 Loss Attribution Result

## Immutable evidence

- Diagnosis workflow run: `30624371230`
- Job: `91136085932`
- Source V20 run: `30623421294`
- Artifact: `8790726669`
- Artifact digest:
  `sha256:5cb3cf1c23b8cac8b8053dd64f7d01f2a3659a27650c313d34a2b0d1e40ccbc6`
- Status: completed successfully
- Successor policy authorized by this diagnosis: no

## Overall V20 result

| Metric | Result |
|---|---:|
| Candidates | 302 |
| Candidate days | 192 |
| Win rate | 46.03% |
| Mean net return | +0.0291% |
| Median net return | -0.0597% |
| Profit Factor | 1.0334 |
| Additional 50bp stress | -0.4709% |
| Return p10 | -2.4966% |

## Loss concentration

- Total negative return points: 263.7725
- Worst 10 candidates: 28.56% of all negative return points
- Worst 20 candidates: 43.74% of all negative return points

Losses are not explained by one or two isolated outliers. Removing a small
number of bad observations would not repair the strategy.

## Model discrimination

The strongest absolute Spearman relationship with realized return was only
0.1262. More importantly:

- V20 expected net return versus realized return: 0.0040
- V20 composite score versus realized return: -0.0161
- V20 stock score versus realized return: -0.0007
- V20 conservative positive probability versus realized return: -0.0383

The hierarchical gate did not produce a stable ordering of future returns.
Continuing to tune its score weights or threshold would be unsupported.

## Exploratory groups

Some individual post-result groups were positive, including the 14:50 slot and
the highest source probability quintile. These are diagnostics, not policies:
they were observed after results, contain limited samples, and were not
independently confirmed. No group is authorized for production or shadow use.

## Decision

V20 remains rejected. The evidence justifies changing the mechanism, not
repairing V20 after the fact.

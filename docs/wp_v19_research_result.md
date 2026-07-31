# WP V19 Broad-Recall Research Result

## Decision

**Rejected. Do not promote to shadow or production.**

V19 widened the causal retrieval frontier but did not find a policy that
passed the preregistered design gates. The nested out-of-sample stream contains
zero authorized candidates because every policy was rejected before test use.
This is not evidence of zero risk or zero losses.

## Immutable Evidence

- Source run: `30600193544`
- Recovery research run: `30619798087`
- Research job: `91121384629`
- Evidence artifact: `8789391249`
- Artifact digest:
  `sha256:712120009e657ca11cb9da2a0a85e837681d344da20d7615245a91b97a65f8f5`
- Evaluation window: `20230727` through `20260724`
- Source rows in the evaluation window: `1,551,890`
- Broad-recall frontier rows: `913,328` (`58.85%` of source rows)
- Full immutable source folds: `1-22`
- Evaluation folds: `5-22`
- Source integrity: passed
- Temporal integrity: passed

## Nested Result

No policy passed its prior design evidence, so no policy was allowed to emit
outer-test candidates:

- Authorized candidates: `0`
- Candidate days: `0`
- Production authorized: `false`
- Future shadow authorized: `false`

The strongest final-design diagnostics were still materially negative:

| Attempt | Events | Candidate days | Win rate | Mean net return | Profit factor | 50bp stress |
|---|---:|---:|---:|---:|---:|---:|
| 1 candidate/day | 5 | 5 | 0.00% | -1.2472% | 0.0000 | -1.7472% |
| Up to 2/day | 10 | 5 | 20.00% | -1.1811% | 0.2403 | -1.6811% |

For the two-candidate attempt, the 10th-percentile return was `-3.2106%`.
None of the 16 preregistered policies passed design; confirmation therefore
was not run.

## Interpretation

The bottleneck was not a narrow candidate pool. Broad retrieval retained more
than nine hundred thousand causal rows and recovered roughly half to seventy
percent of positive source rows by shard. Even so, the stock-level selector
plus absolute probability, return, stability, fill and risk floors produced
too few eligible days, and the few resulting design candidates lost money.

The correct action is not to lower V19 thresholds after seeing this result.
That would be an outcome-driven rule change on an already reused historical
sample.

## Next Independent Hypothesis

V20 will separate two decisions that V19 combined:

1. Is the current day and tail slot a favorable overnight opportunity regime?
2. Which executable stock is best within that regime?

The opportunity gate will be trained only from prior out-of-sample stock
leader streams. Its policy family and evidence gates must be frozen before
running V20. A historically positive result may qualify only for future
150-trading-day shadow validation; it cannot authorize production.

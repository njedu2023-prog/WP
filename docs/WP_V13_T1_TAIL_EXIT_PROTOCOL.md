# WP V13 T+1 Tail Exit Timing Protocol

## Question

Can a fixed, executable T+1 tail sell time turn the immutable V10 entry
candidates profitable after costs?

## Immutable inputs

- V9 causal panel cache from run `30466227350`.
- V10 selected candidates from run `30516136872`.
- V11 candidate frontier from run `30545808015`, verified by SHA-256.

The T-day entry identities, signal slots, entry prices, and entry fill outcomes
are not changed.

## Exit contracts

V13 evaluates four fixed T+1 decisions:

- Decide at 14:20 and execute at the 14:25 five-minute close.
- Decide at 14:30 and execute at the 14:35 five-minute close.
- Decide at 14:40 and execute at the 14:45 five-minute close.
- Decide at 14:50 and execute at the 14:55 five-minute close.

Each execution price receives 10bp adverse sell slippage. The existing 25bp
round-trip cost is then deducted. A benchmark bar must have at least RMB 3m
turnover, and a RMB 100k order must be no more than 1% of that bar. A sell at a
down-limit queue is not credited and receives the predeclared -10% failed-exit
penalty. A target date outside the stored panel is excluded instead of being
fabricated as a loss.

Exit prices and liquidity come from the unfiltered historical five-minute
partitions. They do not come from the decision panel, because that panel is
intentionally filtered by buy eligibility and is therefore not a valid source
for determining whether an existing T+1 holding can be sold.

## Evaluation

V13 reports:

- The full immutable causal candidate frontier.
- The exact 28 V10 selected candidates.
- Paired return deltas against the T+1 close-auction contract.
- Day-clustered intervals for the paired deltas.
- Results by original T-day entry signal slot.
- Baseline and 50bp stress results.

Testing four exit times creates multiple hypotheses. A positive historical mean
is therefore only a research direction. It cannot alter production until it
passes a new frozen 150-trading-day shadow run and all existing promotion gates.

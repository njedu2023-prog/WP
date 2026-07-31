# WP V25 Positioning Dataset Protocol

## Status

The V25 feasibility probe passed before this build was specified. This build
reads the immutable, outcome-blind V24 top-five candidate index and attaches
only data available before each candidate's T-day tail signal.

It must not read candidate returns or authorize a strategy.

## Source

- V24 point-in-time data run: `30635569735`
- V24 candidate rows: `31,955`
- V24 trade dates: `913`
- Candidate identity: fixed V9 broad-recall top five per tail slot
- Existing one-minute, opening-auction, and lagged-money-flow features remain
  immutable.

## New Data

### Previous-Day Holder Cost Distribution

For every candidate stock, query `cyq_perf` once over its required historical
range and retain only immediately previous-trading-day rows.

Derived features compare the immutable T-day signal price with the previous
day's weighted cost and 5/15/50/85/95 percentile costs. They also describe
winner rate, cost-band width, upper supply overhang, and profit cushion.

Minimum candidate-row availability: `95%`.

### Previous-Day Margin Positioning

Query `margin_detail` by required date. Use T-1 values and T-2 values only to
derive financing/lending balances, flow imbalance, and one-day balance
changes.

Stocks outside margin eligibility remain missing with an explicit availability
flag; their missing values are not imputed during construction.

Minimum candidate-row availability: `65%`.

### Previous-Day Abnormal-Trading Disclosure

Query `top_list` by T-1 date. Multiple disclosures for a stock are aggregated
before joining. Absence is a valid zero event, not missing data.

## Integrity

- Candidate identities and V24 features cannot change.
- Every new join is keyed by candidate code and the prior A-share trading date.
- Duplicate stock-date rows in holder-cost or margin sources fail the build.
- No T-day close, T+1 data, return, label, truth, or outcome-driven row removal
  is allowed.
- Mandatory query failures or coverage below the fixed floors reject the
  dataset.

Passing this build authorizes only a separately preregistered nested
out-of-sample V25 study.


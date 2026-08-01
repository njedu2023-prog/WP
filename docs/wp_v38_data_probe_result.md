# WP V38 Index-Futures Data Probe Result

## Decision

V38 is closed at the frozen data-feasibility gate. No candidate profit outcome
was read, no model was fitted, and no profitability conclusion was drawn.

## Immutable Evidence

- workflow run: `30686107050`;
- diagnostic workflow run: `30686376367`;
- all 32 `fut_mapping` queries succeeded;
- all 32 `etf_mins` queries failed because the Tushare account has no
  `etf_mins` access permission;
- all 32 `ft_mins` queries failed because the Tushare account has no
  `ft_mins` access permission;
- total historical minute failures: 64 of 64;
- probe artifact: `wp-v38-index-futures-probe-30686107050`;
- artifact id: `8813989236`;
- artifact digest:
  `sha256:5b35a105798ced0e5259b0da92def57b4eca0d9d3001c92fc5549fd9c79d67bc`.

## Interpretation

This is a source-access failure, not evidence that the derivatives-risk
hypothesis is profitable or unprofitable. The frozen V38 contract explicitly
forbids substituting daily bars, forward-filling missing minutes, weakening
coverage gates, or joining outcomes after a failed data probe.

The next independent direction is V39: T-1 index-futures and ETF-option daily
structure, audited before any candidate return is read.

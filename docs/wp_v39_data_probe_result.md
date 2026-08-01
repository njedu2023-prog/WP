# WP V39 T-1 Derivatives Daily Probe Result

## Decision

Both preregistered source families passed the outcome-blind data-feasibility
gate. A complete three-year point-in-time build is authorized. Model research
and any profitability claim remain unauthorized until that build is frozen.

## Immutable Evidence

- repository commit: `82619f79bd92452a6c9a142e5df9ff4881935ccc`;
- workflow run: `30686915899`;
- job: `91334445955`;
- artifact: `wp-v39-derivatives-daily-probe-30686915899`;
- artifact id: `8814261676`;
- artifact digest:
  `sha256:f45cf1853ced0702c8128e772fcea14443a4edb4556c6861f01ebdee98c0fe12`.

## Coverage

- probe dates: 8;
- exact preceding A-share trade dates: 8 of 8;
- query failures: 0;
- futures mappings: 32 of 32;
- futures daily rows: 512;
- underlying index rows: 32;
- option contract rows: 12,304;
- option daily rows: 4,724;
- underlying ETF rows: 16.

## Frozen Feature Gates

- T-1 futures features: 34 of 34 complete, finite, and sufficiently varying;
- T-1 option features: 17 of 17 complete, finite, and sufficiently varying;
- target-day derivative fields: none;
- candidate outcomes or labels: none;
- forbidden outcome columns: none.

The next gate is a complete `20230727` through `20260724` outcome-blind build.

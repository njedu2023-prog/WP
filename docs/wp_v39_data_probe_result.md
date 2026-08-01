# WP V39 T-1 Derivatives Daily Probe Result

## Decision

Both preregistered source families passed the outcome-blind data-feasibility
gate and the complete three-year point-in-time build. Preregistered V39 model
research is authorized. No profitability claim is authorized before the
nested out-of-sample result is complete.

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

## Three-Year Build Evidence

- repository commit: `e3c10c7402d0a48b6562315621dbf6886d00bccb`;
- workflow run: `30687183695`;
- job: `91335228478`;
- artifact: `wp-v39-derivatives-data-30687183695`;
- artifact id: `8814417777`;
- artifact digest:
  `sha256:1fad354f4694b5270e7e0f7a229a66538554aab31bda76f46fcd572d229375ed`;
- artifact size: 12,316,625 bytes.

## Three-Year Build Coverage

- target dates: 725 of 725;
- exact preceding A-share trade dates: 725 of 725;
- futures mappings: 2,900 of 2,900;
- futures daily rows: 46,392;
- underlying index rows: 2,900;
- option contract rows: 12,281;
- option daily rows: 437,334;
- underlying ETF rows: 1,450;
- output feature rows: 725;
- futures query failures: 0;
- option query failures: 0;
- futures complete and finite rate: 100%;
- option complete and finite rate: 100%;
- varying futures features: 34 of 34;
- varying option features: 17 of 17.

The next gate is the frozen V39 derivatives-license nested out-of-sample study.

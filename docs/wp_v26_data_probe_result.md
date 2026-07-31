# WP V26 Crowd Confirmation Data Probe Result

## Decision

V26 is rejected under its frozen data-feasibility contract. No V26 model
research or profitability backtest is authorized.

This is a data-contract result, not a trading-model result. The probe did not
read T+1 returns or any profit label.

## Immutable Evidence

- Repository: `njedu2023-prog/WP`
- Branch: `main`
- Frozen probe commit:
  `017c383db93ef8a0dee0063797aa49b5834d51d6`
- Workflow run: `30654575396`
- Job: `91235711633`
- Artifact: `8802783702`
- Artifact name:
  `wp-v26-crowd-confirmation-data-probe-30654575396`
- Artifact digest:
  `sha256:edbdc65cd849c58511f2f7a70b86ad05d3d0312b9254a4b09f663f874342ade3`
- Artifact size: `416,686` bytes
- Contract tests: `10 passed`
- `profit_outcomes_read`: `false`
- `selected_source_family`: `null`
- `full_backfill_authorized`: `false`

## THS Historical Hot List

THS supplied a usable 14:30 batch on every probe date, but only seven of eight
dates passed the frozen schema contract.

| Probe date | Tail rows | Unique A-share codes | Raw rank range | Pass |
| --- | ---: | ---: | ---: | :---: |
| 2023-08-25 | 100 | 100 | 2,551,553-27,034,057 | no |
| 2023-12-29 | 100 | 100 | 1-100 | yes |
| 2024-03-15 | 100 | 100 | 1-100 | yes |
| 2024-09-27 | 100 | 100 | 1-100 | yes |
| 2025-01-15 | 100 | 100 | 1-100 | yes |
| 2025-07-23 | 99 | 99 | 1-100 | yes |
| 2026-01-15 | 100 | 100 | 1-100 | yes |
| 2026-07-23 | 99 | 99 | 1-100 | yes |

The 2023-08-25 `rank` field is not an ordinal rank. It is in the millions and
also fails the frozen date-consistency check. V26 did not reinterpret the
field after seeing this result.

## DC Historical Hot List

DC passed five of eight probe dates after duplicate same-minute fetches were
correctly separated. It returned zero rows for:

- 2023-08-25
- 2023-12-29
- 2024-03-15

DC therefore cannot support the required three-year evaluation window.

## Fine Industry Fallback

The static fine-industry metadata was complete:

- published SW2021 L2 indices: `124`
- published SW2021 L3 indices: `259`
- historical membership rows: `7,943`
- active-stock counts across probe dates: `5,509-5,880`
- L2 and L3 membership coverage: `100%`

However, all `64 / 64` sampled `sw_mins` requests failed because the configured
Tushare account does not have permission for the separately licensed SW
historical-minute endpoint. The fallback therefore failed as an executable
data source.

## Closed Decisions

- Do not run a V26 profitability study.
- Do not lower the eight-date coverage rule.
- Do not use DC as a three-year source.
- Do not silently reinterpret the legacy THS `rank` values inside V26.
- Do not claim that the successful workflow means the data passed.

A separate outcome-blind protocol may investigate whether the legacy THS
schema can be normalized using contemporaneous source fields. That
normalization must be frozen before any T+1 outcome is read.

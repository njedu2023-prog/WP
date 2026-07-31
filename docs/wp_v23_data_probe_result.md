# WP V23 Point-in-Time Data Probe Result

## Decision

The V23 point-in-time data feasibility probe passed. Targeted historical
dataset construction is authorized. This result does not authorize a strategy,
shadow deployment, or production use.

The probe intentionally read no profit outcomes. It tested only whether the
new data families are available, point-in-time usable, and structurally
complete enough to justify a preregistered research build.

## Immutable Evidence

- GitHub Actions run: `30626970634`
- Job: `91144375424`
- Artifact: `8791788839`
- Artifact digest:
  `sha256:db3e8d3a2d3750b5d5193e17f7848fdb902d121b503677f182c971a3e3708b1c`
- Probe contract: `WP_V23_DATA_FEASIBILITY_PROTOCOL.md`
- `profit_outcomes_read`: `false`
- `v23_backfill_authorized`: `true`

## Historical One-Minute Bars

The probe covered four dates spanning 2023, 2024, 2025, and 2026 and three
liquid main-board stocks from both exchanges.

- Symbol-date probes: `12 / 12` passed
- Rows in the requested 13:55-15:00 interval: `66` per probe
- Required 14:01-15:00 rows: `60 / 60` per probe
- Tail coverage: `100%`
- Duplicate timestamps: none
- OHLC consistency: passed
- Positive volume and amount coverage: `96.97%-98.48%`

The five-minute controls also passed with `12 / 12` required tail bars.

## Opening Auction

The same-day opening-auction cross section passed on all three sampled dates.

| Trade date | Rows | Unique codes | Duplicate stock-date |
| --- | ---: | ---: | --- |
| 2024-01-02 | 5,319 | 5,319 | No |
| 2025-01-16 | 5,374 | 5,374 | No |
| 2026-07-24 | 5,509 | 5,509 | No |

All sampled stocks were present and all requested numeric fields had complete
coverage.

## Previous-Day L2 Money Flow

The daily L2 money-flow cross section passed on all four sampled prior trading
dates.

| Prior trade date | Rows | Unique codes | Duplicate stock-date |
| --- | ---: | ---: | --- |
| 2023-08-25 | 5,045 | 5,045 | No |
| 2023-12-29 | 5,088 | 5,088 | No |
| 2025-01-15 | 5,115 | 5,115 | No |
| 2026-07-23 | 5,198 | 5,198 | No |

All sampled stocks were present and all requested numeric fields had complete
coverage. Research use is restricted to the immediately preceding A-share
trading date.

## Authorized Next Step

V23 may now construct a targeted, immutable dataset for the full V9
out-of-sample opportunity set:

1. Select required stock-month pairs without reading outcomes.
2. Fetch one-minute bars only for those fixed pairs.
3. Join same-day completed opening-auction data.
4. Join only previous-trading-day L2 money-flow data.
5. Compute every signal feature using bars at or before the signal slot.
6. Fail closed on missing coverage; never drop observations based on returns.
7. Freeze the V23 model, policy, and gates before reading V23 results.

Only a preregistered nested out-of-sample pass can authorize the required
150-trading-day future shadow run.

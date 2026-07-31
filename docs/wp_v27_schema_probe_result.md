# WP V27 THS Historical Schema Probe Result

## Decision

V27 is rejected. The historical THS hot-list schema cannot be normalized over
the fixed three-year sample without guessing.

## Immutable Evidence

- Workflow run: `30655157656`
- Job: `91237623053`
- Source commit: `fa8bdb50934cb9ce843994c127655e034d2291b4`
- Probe dates: `12`
- Passed dates: `5`
- Full backfill authorized: `false`
- Profit outcomes read: `false`

## Failure

- `2023-07-28` and `2023-11-30` had no usable tail snapshot.
- `2023-08-25`, `2023-09-28`, and `2023-10-31` used a large-valued legacy
  `rank` field but supplied no contemporaneous `hot` value to establish its
  direction.
- `2025-07-23` and `2026-07-23` each contained only 99 valid A-share codes in
  the selected batch.

Response order and unconfirmed score direction are forbidden reconstruction
methods. V27 therefore stops before any profitability outcome is read.

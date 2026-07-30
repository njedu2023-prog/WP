# WP V18 research result

## Immutable evidence

- Source commit: `e72087bc78a8d8f36498531bacca9850517c772a`
- Workflow run: `30570688654`
- Workflow job: `90966351715`
- Evidence artifact: `8771361279`
- Artifact SHA-256:
  `37053af3cc81f7d1ec80eb6a31daa01924a526dfd3d251d0cacdf431e0579f96`
- Frozen V9 fingerprint: `16695b7fab38a5428f70`
- Immutable V11 source run: `30545808015`
- Frozen V15 source run: `30552732652`
- Exit contract: fixed T+1 close after established round-trip costs

## Protocol result

The remote workflow completed successfully. Frozen-contract tests, temporal
ordering checks, nested model fitting, policy calibration, and artifact upload
all passed operationally.

The final result is `NOT_READY_FOR_SHADOW`:

- Nested authorized candidates: `0`
- Nested authorized candidate days: `0`
- Final policy: none
- Design policies passing all predeclared gates: `0 / 16`
- Confirmation: not run because no design policy was eligible
- Temporal integrity: passed
- Production authorization: false

The strongest final design-window policy was positive but too small to
authorize:

- Events: `5`
- Candidate days: `4 / 42` (`9.52%`)
- Win rate: `80.00%`
- Mean net return: `+2.9130%`
- Mean net return after an additional 50 bps cost: `+2.4130%`
- Profit factor: `595.66`
- BH-adjusted q-value: `0.09434`

It failed the predeclared minimums of 10 events and 6 candidate days.
Confirmation therefore remained untouched.

## Descriptive signal, not authorization

Pooling the final design and confirmation periods for diagnostics only produced
a practical-frequency frontier:

- Events: `10`
- Candidate days: `7 / 84` (`8.33%`)
- Win rate: `80.00%`
- Mean net return: `+2.9849%`
- Day-clustered mean lower bound: `+1.2422%`
- Mean net return after an additional 50 bps cost: `+2.4849%`
- Profit factor: `6.9519`
- Return 10th percentile: `-0.5211%`
- BH-adjusted q-value: `0.005596`

This pooled diagnostic is encouraging, but it is not an independent
confirmation result and must not be used to authorize production or a public
buy list.

## Decision

V15 remains frozen and unpromoted. V18 remains research-only. No threshold is
to be relaxed after seeing this result, and the same historical holdout must
not be relabeled as fresh confirmation evidence.

The next admissible evidence is untouched future telemetry. Any successor
policy must be preregistered before that evidence is observed and must still
complete at least 150 future A-share trading days plus the required candidate
and candidate-day counts before production can be considered.

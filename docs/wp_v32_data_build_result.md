# WP V32 Public-Event Data Build Result

## Decision

The preregistered V32 three-year outcome-blind public-event dataset passed
every construction, coverage, identity, and probe-parity gate. The single
frozen V32 nested out-of-sample model study is authorized.

This result establishes only that the four admitted public-event families can
be joined causally and at useful coverage to the immutable candidate set. It
does not establish predictive value and does not authorize shadow deployment,
public candidates, or production use.

## Immutable Evidence

- Protocol and implementation commit:
  `683332034e24bf187130bdb7a038b60bcc76a058`
- Immutable V9 candidate source run: `30600193544`
- Immutable V24 point-in-time source run: `30635569735`
- V32 data build run: `30665412715`
- V32 data build job: `91271145335`
- Artifact: `8808328109`
- Artifact name: `wp-v32-public-event-data-30665412715`
- Artifact digest:
  `sha256:93ca2d037905e681cdf96529f65599c367b9dc224c41438d4480bed7db671eed`
- Artifact size: `458,970` bytes
- Outcome-blind candidate rows: `31,955`
- Feature rows: `31,955`
- Source trade dates: `913`
- Required source dates: `917`
- Exact event queries: `3,668`
- Query failures: `0`
- `v32_model_research_authorized`: `true`

## Coverage Result

| Measure | Result | Frozen gate | Passed |
| --- | ---: | ---: | --- |
| Candidate identity match | exact | exact | yes |
| Complete common fields | 31,955 / 31,955 | 100% | yes |
| Active-flag consistency | exact | exact | yes |
| Minimum active-detail coverage | 98.6072% | at least 80% | yes |
| Event-active candidate rows | 5,232 / 31,955 | at least 5% | yes |
| Event-active candidate row rate | 16.3730% | at least 5% | yes |
| Event-active trade dates | 807 / 913 | at least 50% | yes |
| Event-active trade-date rate | 88.3899% | at least 50% | yes |
| Probe identity parity | 121 / 121 | exact | yes |
| Probe presence mismatches | 0 | 0 | yes |

## Source Coverage

| Source | Active rows | Active-detail coverage |
| --- | ---: | ---: |
| `forecast` | 426 | 99.2958% |
| `repurchase` | 718 | 98.6072% |
| `share_float` | 290 | 100.0000% |
| `block_trade` | 4,021 | 100.0000% |

The event union is smaller than the sum because a candidate can have more than
one prior-five-trading-day event source.

## Integrity Controls

- Candidate identities were frozen before the build and candidate outcomes
  were not read.
- Every event query used exact prior A-share trading dates and retained only
  normalized A-share security codes.
- The feature window contains only the five immediately prior trading days.
- All `3,668` exact source-date queries completed without a recorded failure.
- Inactive rows remain explicit causal baselines rather than being deleted.
- Missing event detail remains missing and is never inferred from later data.
- Full-build identities and event-presence flags reproduce the frozen probe
  sample exactly.

## Authorized Next Step

Run exactly one V32 nested out-of-sample study under
`WP_V32_PUBLIC_EVENT_MODEL_PROTOCOL.md`. No feature, label, threshold, model,
candidate cap, acceptance gate, or exit-contract change is allowed after the
first result is read.

Even a complete historical pass can authorize only an unchanged 150 future
A-share trading-day shadow run. It cannot establish production profitability.

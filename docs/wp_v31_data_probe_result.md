# WP V31 Public-Event Data Probe Result

## Decision

V31 is closed at the frozen data-feasibility gate.

It is not authorized for a three-year backfill, model research, shadow
evaluation, or production use. No profit outcome was read during source
selection or probing.

The failure is attributed to a preregistered universe-audit error, not to
absence of candidate coverage. A separate V32 protocol may re-audit the same
immutable raw responses after excluding non-A-share rows. V31 itself will not
be changed.

## Immutable Evidence

- Frozen implementation commit:
  `75d399accaec69da3c151446bb3570197eaecd21`
- Workflow run: `30663984930`
- Workflow job: `91266507635`
- Artifact: `wp-v31-public-event-data-probe-30663984930`
- Artifact ID: `8806378449`
- Artifact digest:
  `sha256:11f60e1b87609524cae09405221a6c096eec9c3319e5b550ba87f69188b8fec4`
- Immutable V24 outcome-blind candidate source run: `30635569735`

The workflow conclusion is intentionally `failure`: the probe wrote its
evidence and then raised because the frozen family gate did not pass.

## Probe Result

- Probe dates: `8`
- Required prior trading dates: `40`
- Source-date queries: `240`
- Candidate identities: `121`
- Sources admitted under the V31 contract: `forecast`
- Matched candidate identities from admitted sources: `3`
- Matched target dates from admitted sources: `2`
- Full backfill authorized: `false`
- Next gate: `close_v31_data_direction`

## Failure Attribution

Three otherwise useful sources failed because a raw response contained at
least one code outside the A-share universe:

| Source | Candidate matches | Target dates | V31 query contract |
|---|---:|---:|---|
| `repurchase` | 3 | 3 | failed |
| `share_float` | 3 | 3 | failed |
| `block_trade` | 14 | 6 | failed |

The frozen V31 audit required every raw row returned by a market-wide
interface to be an A-share code. That assumption is invalid for interfaces
which can also return other listed security universes. Nine of 240 queries
failed only this code-domain check. Their schemas and requested dates passed.

This matters because `block_trade` alone already matched 14 candidate
identities across six probe target dates, above the frozen family-level
requirements of ten identities and five target dates. The data family
therefore has enough observed coverage to justify a clean re-audit, but V31
cannot be rescued after seeing the result.

## Frozen Conclusion

V31 is a failed probe contract, not a profitable-factor result. It did not
read returns and says nothing about predictive value.

The following are prohibited under V31:

- changing its raw-code requirement in place;
- treating the 14 block-trade matches as evidence of positive return;
- using V31 to authorize a full backfill;
- relaxing the original source or family coverage thresholds;
- reading candidate outcomes before the corrected data gate passes.

V32 must use the same raw artifact, candidate source, dates, lookback,
sources, and thresholds. Its only allowed change is explicit A-share universe
normalization before source-level auditing and candidate joins.

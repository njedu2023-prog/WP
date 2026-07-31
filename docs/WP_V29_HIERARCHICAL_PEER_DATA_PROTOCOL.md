# WP V29 Hierarchical Peer Data Protocol

## Research Boundary

V28 is closed because only 96.9707% of candidates met its frozen simultaneous
L2 and L3 peer-depth contract. The outcome-blind diagnosis proved all 968
incomplete rows were caused by genuinely shallow peer groups, not engineering
missingness.

V29 defines a new representation before reading any profitability outcome. It
does not change V28 and does not authorize a model by itself.

## Immutable Inputs

- V24 candidate identities and trade calendar from data run `30635569735`;
- V28 leave-one-out L2 and L3 features from data run `30656696310`;
- V28 outcome-blind coverage diagnosis from run `30659154353`;
- exactly 31,955 immutable date-slot-stock identities;
- fixed signal slots 14:20 through 14:50 at five-minute intervals.

All source manifests and parquet files must pass SHA-256 checks. V28's
candidate-excluded peer aggregates remain immutable.

## Fixed Hierarchical Transformation

For each metric available at both L2 and L3, let `n3` be the leave-one-out L3
peer count. The fixed fine-industry weight is:

```text
w3 = n3 / (n3 + 6)
hierarchical_metric = w3 * L3_metric + (1 - w3) * L2_metric
```

The pseudo-count `6` is fixed without outcomes and cannot be tuned later.

- when L3 is unavailable and L2 exists, use L2;
- when L2 is unavailable and L3 exists, use L3;
- when both are unavailable, store numeric zero plus an explicit
  `no_peer_context` indicator;
- retain L2/L3 counts, availability flags, shallow-depth flags, L3 weight, and
  L3-to-L2 depth ratio;
- never drop a candidate because an industry group is small.

The zero fallback is only a finite storage value. The paired availability
indicator is mandatory so a model cannot confuse missing peer context with a
measured neutral peer result.

## Data Acceptance Gate

The V29 output must satisfy all of the following:

- exact identity and fold match to the immutable V24/V28 candidate set;
- no duplicate date-slot-stock identity;
- zero candidate rows dropped;
- every V29 model feature finite on 100% of candidate rows;
- source and diagnosis digests match;
- `profit_outcomes_read=false`;
- `future_information_allowed=false`.

Only a pass authorizes a separate commit that preregisters one V29 nested
out-of-sample model and release policy. A data pass is not evidence of profit
and does not authorize shadow or production use.

# WP V33 Research Recovery

## Incident

- Failed workflow run: `30675048976`.
- Failure point: first outer-fold policy calibration.
- No outer-fold prediction, economic result, or `WP_V33_RESULT` was produced.
- The immutable V33 data artifact was not changed.

The policy frame referenced four V23 model-disagreement columns that are not
part of the immutable V24 point-in-time source:

- `v23_expected_return_model_spread_pct`
- `v23_margin_model_spread`
- `v23_positive_model_spread`
- `v23_severe_model_spread`

Synthetic policy tests had supplied those fields and therefore did not expose
the source-contract mismatch.

## Recovery

The four unsupported external gates were removed before any valid outer-fold
result existed. This is an engineering source-contract correction, not
post-result model selection.

The following remain unchanged:

- V33 feature family and all model hyperparameters;
- train, purge, calibration, and outer-test windows;
- calibration-derived score-threshold rule;
- V33 positive-return, severe-loss, expected-return, and pairwise gates;
- source fill, source severe-risk, and strict data-age gates;
- execution, cost, stress, frequency, rank, and historical pass gates;
- all-qualified, no-daily-cap publication policy.

The recovery workflow is the first run permitted to produce a valid V33
outer-fold result. Any failed preregistered gate still rejects V33 and forbids
threshold repair.

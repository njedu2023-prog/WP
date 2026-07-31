# WP V28 Data Diagnosis Result

## Decision

V28 is closed at the data gate. Its frozen 98% complete-feature threshold is
not changed after the result.

The outcome-blind diagnosis ran in GitHub Actions run `30659154353`, job
`91250921216`, against the immutable V28 data artifact from run
`30656696310`.

## Evidence

- candidate rows: 31,955;
- complete rows: 30,987;
- complete coverage: 96.9707%;
- incomplete rows: 968 across 377 trade dates;
- rows with insufficient L2 peer depth: 353;
- rows with insufficient L3 peer depth: 683;
- rows with either peer level too shallow: 968;
- incomplete rows despite sufficient peer depth: 0;
- amount-only missing rows despite sufficient peer depth: 0.

The shortfall is therefore not a query, timestamp, join, or amount-field
engineering defect. It is a real statistical support problem: some historical
fine-industry groups contain too few leave-one-out peers.

## Next Gate

V29 is a separately preregistered data representation. It may shrink fine
industry evidence toward the broader industry and preserve explicit
availability indicators. It may not drop candidates, lower V28's threshold,
or inspect T+1 outcomes while defining the transformation.

# WP V33 Three-Year Data Build Result

## Decision

The frozen outcome-blind V33 data contract passed. The preregistered
limit-industry ecology model research is authorized. This result contains no
candidate profit outcome and does not authorize production.

## Immutable Evidence

- Repository: `njedu2023-prog/WP`
- Data workflow run: `30671468932`
- Source commit: `b5af5f2ed6399088ac55003d34a658ebd3ec8859`
- Artifact: `wp-v33-limit-industry-data-30671468932`
- Artifact ID: `8810059603`
- Artifact digest:
  `sha256:aeec52cbcebe1cd055ed0efdb495e61b882e67c72aa5d43b89a6ace1c9b970f7`

## Frozen Contract Results

- Candidate rows: `31,955`
- Feature rows: `31,955`
- Candidate trading dates: `913`
- Required source dates: `914`
- Source queries: `2,742`
- Query failures: `0`
- Query contract: passed
- Date contract: passed
- Candidate identity match: passed
- Numeric feature completeness: passed
- Candidate industry membership: `99.9531%`
- Current event industry membership: `99.9421%`
- Previous-day event industry membership: `99.9421%`
- Same-L2 active rows: `10,399` (`32.5426%`)
- Same-L2 active dates: `858 / 913` (`93.9759%`)
- Same-L3 active rows: `5,731` (`17.9346%`)
- Same-L3 active dates: `771 / 913` (`84.4469%`)
- Full three-year coverage: passed
- Probe feature parity: `280 / 280`
- Probe mismatch rows: `0`
- Maximum probe absolute difference: `0`
- Profit outcomes read during build: no
- Future information allowed: no

## Next Gate

Freeze the single V33 model and policy before reading outcome shards, then run
one nested out-of-sample test. Failed economic or rank gates reject V33 without
post-result threshold repair. A historical pass can authorize only an
unchanged 150-future-trading-day shadow run.

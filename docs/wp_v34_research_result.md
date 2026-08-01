# WP V34 Full-Session Path Research Result

## Decision

V34 is rejected. The preregistered full-session intraday-path model failed the
historical nested out-of-sample economic, frequency, tail-risk, and
same-slot-ranking gates. It is not authorized for the 150-trading-day shadow
run or production.

This is a terminal conclusion for the frozen V34 protocol. Do not retune its
thresholds, remove losing observations, invert outputs, or promote isolated
folds or years after reading this result.

## Immutable Evidence

- Full-data implementation commit:
  `c44264d67e1c23ac0886f1a1b77db3850c51930d`
- Frozen model protocol commit:
  `77e71e7aa0f80a4fc7349f3d0660fe1846b5264c`
- Frozen research implementation commit:
  `5e51f05f912718662f7b88f8472c7b474d994b3d`
- Research trigger commit:
  `af3ed1d1c563549c882bcd5d43c2cf5c86860dd6`
- Immutable V9 source run: `30600193544`
- Immutable V24 candidate-data run: `30635569735`
- V34 full-data run: `30677075531`
- V34 full-data job: `91306460434`
- V34 full-data artifact: `8812271933`
- V34 full-data artifact digest:
  `sha256:9188ab5442fce82fadf0ec736243ec58d7c10e36ef9a99e54947ac9cd58758dd`
- V34 research run: `30681611099`
- V34 research job: `91319614595`
- V34 research artifact: `8812555077`
- V34 research artifact digest:
  `sha256:3098dfc33ad58225ddcd971d537e3dcec39f54fc4c601d64818e82123f8a1841`

## Data Construction

- Outcome-blind source candidates: `31,955`
- Joined research rows: `31,955`
- Source trade dates: `913`
- Evaluation days: `725`
- Historical one-minute rows: `3,830,456`
- Required stock-month queries: `9,669`
- Query failures: `0`
- Candidate coverage: `100%`
- Signal-price parity: `100%`
- Finite-feature coverage: `100%`
- Causal timestamp audit: passed
- Probe identities, numeric features, and quality fields: exact match
- Forbidden future columns: none

## Frozen Nested OOS Result

| Metric | Result | Frozen requirement |
| --- | ---: | ---: |
| Candidates | `25` | at least `120` |
| Candidate days | `17` | at least `80` |
| Candidate-day rate | `2.3448%` | `12%` to `35%` |
| Win rate | `52.00%` | at least `55%` |
| Wilson win-rate lower bound | `33.50%` | at least `50%` |
| Clustered win-rate lower bound | `34.62%` | at least `48%` |
| Mean net return | `-0.5091%` | at least `+0.20%` |
| Median net return | `+0.0157%` | diagnostic only |
| Clustered mean lower bound | `-1.1389%` | above `0%` |
| Profit Factor | `0.6975` | at least `1.20` |
| Additional 50bp stress mean | `-1.0091%` | at least `0%` |
| Return p10 | `-5.3713%` | at least `-3%` |
| Margin-hit rate | `48.00%` | at least `20%` |
| Tail-loss rate | `32.00%` | no more than `25%` |

The mean-return p-value was `0.5705`; the observed mean is both economically
negative and statistically unsupported.

## Same-Slot Ranking

- Same-day same-slot groups: `5,075`
- Mean within-slot rank IC: `-0.01649`
- Mean top-minus-bottom return spread: `-0.01748%`
- Clustered spread interval: `[-0.13611%, +0.09808%]`

The new full-session path family therefore did not improve the ability to
choose a better stock from the candidates available at the same decision
time.

## Calendar Stability

| Year | Candidates | Candidate days | Mean net return |
| --- | ---: | ---: | ---: |
| 2023 | `1` | `1` | `+1.4559%` |
| 2024 | `14` | `8` | `-0.3750%` |
| 2025 | `1` | `1` | `+1.0553%` |
| 2026 | `9` | `7` | `-1.1097%` |

The apparently positive years contain one candidate each and provide no
usable stability evidence. The two years with meaningful counts are negative.

## Gate Verdict

Fifteen historical gates failed. Only the margin-hit floor, active-year count,
and temporal/source/data integrity gates passed. All selected outcomes were
present and internally consistent, so the rejection is an economic result,
not a missing-data artifact.

## Authorized Next Step

Do not deploy or shadow V34. A successor must be frozen as a new hypothesis
before outcomes are read and must address the observed structural failure:
the current candidate features do not provide positive same-slot stock
selection and the strict policy produces too few, too risky trades.

The next admissible direction is a hierarchical regime-first study: predict
whether a tail-session market regime is tradable using only point-in-time
cross-sectional market information, then apply a separately frozen
stock-level risk screen inside accepted regimes. This must be evaluated with
the same executable entry, fixed T+1 close exit, costs, no-trade option,
nested out-of-sample process, and 150-future-day shadow gate.

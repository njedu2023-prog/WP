# WP V33 Limit-Industry Ecology Data Feasibility Protocol

## Status

This outcome-blind protocol is frozen before any V33 candidate return is read.
A pass authorizes only a full three-year point-in-time data build. It cannot
authorize model research, shadow operation, public candidates, production, or
a profitability claim.

## Independent Hypothesis

V30 proved that the historical `kpl_list` tape contains a dense market-wide
record of limit-up touches, opens after a touch, and limit-down touches by the
14:20-14:50 decision slots. V30 failed because none of its sampled candidates
was itself a limit-event stock.

V33 asks a different question without using outcomes: can those already
observed events be mapped through historical SW2021 membership to each
candidate's L2 and L3 industry with enough coverage to support a genuine
same-slot stock-selection study?

The mechanism is industry diffusion, not the candidate's own limit event.
Every industry aggregate excludes the candidate itself.

## Immutable Inputs

- V24 outcome-blind candidate source run: `30635569735`;
- V28 historical industry-membership source run: `30656696310`;
- V30 outcome-blind market-projection probe run: `30662958173`;
- candidate probe dates:
  - `20230825`;
  - `20231229`;
  - `20240315`;
  - `20240927`;
  - `20250115`;
  - `20250723`;
  - `20260115`;
  - `20260723`;
- signal slots: 14:20 through 14:50 at five-minute intervals;
- current and immediately previous A-share trading dates from the immutable
  V24 exchange calendar.

## Causal Source Contract

For each required date, query `kpl_list` independently for `涨停`, `炸板`, and
`跌停`. Current-day features may use only:

- stock identity;
- first limit-up touch time;
- first open time after a touch;
- first limit-down touch time;
- events whose timestamp is at or before the decision slot.

The union of final `涨停` and `炸板` categories may recover stocks that touched
the up limit, but their final category may not become a current-day feature.
Post-14:50 events are ignored for every live candidate feature. `last_time`,
final status, text reasons, themes, end-of-day prices, turnover, amounts, and
order values are forbidden current-day outputs.

The immediately previous trading day is complete information. Its event counts
may be used only after mapping through the membership active on that previous
date.

## Frozen Candidate Features

For market-wide, L2-industry, and L3-industry scopes:

- cumulative limit-up touches by the slot;
- cumulative opens after a touch by the slot;
- cumulative limit-down touches by the slot;
- new touches and opens in the prior ten minutes;
- net sealed count;
- open-to-touch ratio.

L2 and L3 features also contain:

- active peer-member count, excluding the candidate;
- touch and down-limit rates per active peer;
- share of all-market limit touches;
- prior-day completed counts and rates.

Raw industry codes are retained only for identity audit. A later model may not
one-hot encode or target-encode raw industry identities.

## Frozen Probe Gates

The family passes only if all gates pass:

1. all required date-category queries succeed under the frozen schema;
2. every supplied first-event and open timestamp parses, and open does not
   precede first touch;
3. every date produces seven unique complete market-slot projections;
4. the eight current-date projections exactly reproduce the immutable V30
   market projection;
5. at least 98% of sampled candidates have active L2 and L3 membership;
6. at least 90% of current and previous event stocks have active L2 and L3
   membership;
7. all sampled candidate feature rows are unique, finite, and complete;
8. at least 20% of candidate rows have a prior same-L2 limit touch by signal;
9. at least 10% have a prior same-L3 limit touch by signal;
10. same-L2 coverage appears on at least six of eight target dates;
11. same-L3 coverage appears on at least four of eight target dates;
12. no forbidden end-of-day field or candidate return is read or emitted.

## Decision Rule

Any failed gate closes V33 before a profitability study. A pass authorizes
only the full three-year outcome-blind build with the same features, dates,
timestamps, exclusions, and coverage gates.

A later model protocol must be frozen in a separate commit after the full data
contract passes. Historical success could authorize only an unchanged
150-future-trading-day shadow run.

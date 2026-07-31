# WP V25 Positioning Data Feasibility Protocol

## Status

This protocol is frozen before any V25 profit outcome is read.

The probe is outcome-blind. Passing it authorizes only construction of a
point-in-time research dataset. It cannot authorize a model, public candidate,
shadow deployment, or production use.

## Motivation

V24 showed that the V9-V24 model outputs mostly distinguish favorable market
dates and do not reliably rank stocks available in the same tail slot. No
existing causal feature produced a stable, economically material same-slot
return spread.

V25 therefore tests information that was not present in the V9-V24 feature
contract:

1. prior-close holder cost distribution and profitable-holder share;
2. prior-close financing and securities-lending positioning;
3. prior-close exchange abnormal-trading disclosures;
4. prior-date company-announcement metadata, if the account has reproducible
   access.

The first two families form the mandatory positioning dataset. The latter two
are optional sparse event families and are admitted independently.

## Causal Availability

- `cyq_perf`: use the immediately previous A-share trading date only. Tushare
  updates it after that close.
- `margin_detail`: use the immediately previous A-share trading date only.
  The exchange publishes the prior date before the next session.
- `top_list`: use the immediately previous A-share trading date only.
- `anns_d`: use prior-date announcements only in the historical build. A
  later live design may admit same-day records only when an immutable
  `rec_time` is no later than the decision cutoff.

No same-day closing value, revised future record, T+1 field, candidate return,
or truth outcome may enter the data probe or dataset selector.

## Frozen Probe

Dates span the three-year evaluation period:

- `20230825`
- `20231229`
- `20250115`
- `20260723`

Long-lived sample stocks cover both exchanges and major boards:

- `600000.SH`
- `000001.SZ`
- `601318.SH`
- `300750.SZ`
- `688981.SH`

### Holder Cost Distribution

- API: `cyq_perf`
- Required fields: historical low/high, 5/15/50/85/95 percentile costs,
  weighted average cost, and winner rate.
- Every symbol-date probe must exist exactly once.
- Cost percentiles must be ordered, weighted cost must be positive, and winner
  rate must be within `[0, 100]`.

### Margin Positioning

- API: `margin_detail`
- Minimum cross-section: `1,500` unique stocks per sampled date.
- Requested financing/lending numeric fields must have at least `95%`
  finite coverage.
- At least four of the five sample stocks must be present on every date.

### Abnormal-Trading Disclosure

- API: `top_list`
- Sparse data are valid and absence for a stock is meaningful.
- Every sampled date must return a valid schema and at least one disclosed
  stock.

### Announcement Metadata

- API: `anns_d`
- This family is optional because access is separately permissioned.
- If accessible, every sampled date must return a valid schema and at least one
  announcement.

## Decision Rule

V25 positioning dataset construction is authorized only when both holder cost
distribution and margin positioning pass every frozen probe. Sparse event
families are admitted only when their own probes pass.

The subsequent historical study, if authorized, must be separately frozen. It
must explicitly test same-day same-slot rank IC and return spread, retain the
fixed executable entry and T+1 close exit, apply all costs, and require 150
untouched future A-share shadow days before any production decision.


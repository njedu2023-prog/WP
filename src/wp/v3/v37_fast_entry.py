from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS
from .v34_intraday_path import normalize_historical_minutes
from .v34_path_ranker import (
    EXPECTED_NET_RETURN_LOWER_MIN_PCT,
    FIXED_MAX_CANDIDATES_PER_DAY,
    MARGIN_PROBABILITY_LOWER_MIN,
    POSITIVE_PROBABILITY_LOWER_MIN,
    PROBABILITY_SPREAD_MAX,
    RETURN_SPREAD_MAX_PCT,
    ROUND_TRIP_FILL_MIN,
    SEVERE_PROBABILITY_UPPER_MAX,
    SOURCE_SEVERE_LOSS_MAX,
    policy_eligible_rows as v34_policy_eligible_rows,
)
from .v36_entry_confirmation import v36_research_readiness


SCHEMA_VERSION = "wp_v37_fast_entry_contract_1"
BASE_ALERT_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
)
PUBLICATION_DELAY_MINUTES = 2
ENTRY_DELAY_MINUTES = 3
MINIMUM_DATASET_COVERAGE = 0.98
FIXED_TARGET_CANDIDATE_DAY_RATE = 0.25

OUTCOME_COLUMNS = (
    "entry_price",
    "gross_return_pct",
    "net_return_pct",
    "entry_fillable",
    "exit_fillable",
    "execution_success",
    "label_available",
    "target_net_positive",
    "target_severe_loss",
)

QUALITY_COLUMNS = (
    "v37_publication_time",
    "v37_entry_benchmark_time",
    "v37_entry_bar_time",
    "v37_entry_bar_close",
    "v37_entry_bar_amount",
    "v37_entry_bar_volume",
    "v37_entry_distance_to_up_limit_pct",
    "v37_entry_distance_to_down_limit_pct",
    "v37_entry_bar_exact",
    "v37_entry_data_complete",
)


@dataclass(frozen=True)
class FastEntryPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"v37-fast-entry-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "publication_delay_minutes": PUBLICATION_DELAY_MINUTES,
            "entry_delay_minutes": ENTRY_DELAY_MINUTES,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "positive_probability_lower_min": (
                POSITIVE_PROBABILITY_LOWER_MIN
            ),
            "margin_probability_lower_min": MARGIN_PROBABILITY_LOWER_MIN,
            "severe_probability_upper_max": (
                SEVERE_PROBABILITY_UPPER_MAX
            ),
            "expected_net_return_lower_min_pct": (
                EXPECTED_NET_RETURN_LOWER_MIN_PCT
            ),
            "probability_spread_max": PROBABILITY_SPREAD_MAX,
            "return_spread_max_pct": RETURN_SPREAD_MAX_PCT,
            "first_qualifying_signal_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenFastEntryPolicy:
    spec: FastEntryPolicySpec
    score_threshold: float
    calibration_start: str
    calibration_end: str
    calibration_days: int
    eligible_days: int

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "score_threshold": self.score_threshold,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "calibration_days": self.calibration_days,
            "eligible_days": self.eligible_days,
        }


def fast_entry_timing(signal_slot: str) -> tuple[str, str]:
    if signal_slot not in BASE_ALERT_SLOTS:
        raise ValueError(f"unsupported V37 base alert slot: {signal_slot}")
    base = pd.Timestamp(f"2000-01-01 {signal_slot}:00")
    publication = base + pd.offsets.Minute(PUBLICATION_DELAY_MINUTES)
    entry = base + pd.offsets.Minute(ENTRY_DELAY_MINUTES)
    return publication.strftime("%H:%M"), entry.strftime("%H:%M")


def build_fast_entry_outcomes(
    candidates: pd.DataFrame,
    minutes: pd.DataFrame,
    *,
    entry_slippage_bps: float,
    round_trip_cost_bps: float,
    min_slot_amount: float,
    reference_order_notional: float,
    max_entry_pct_of_slot_amount: float,
    min_distance_to_up_limit_pct: float,
    min_distance_to_down_limit_pct: float,
    non_fill_penalty_pct: float,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        "execution_eligible",
        "source_label_available",
        "target_trade_date",
        "t1_total_return_close",
        "source_exit_fillable",
        "up_limit",
        "down_limit",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V37 candidates missing columns: {missing}")
    source = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS),
        [
            *IDENTITY_COLUMNS,
            "fold",
            "signal_price",
            "execution_eligible",
            "source_label_available",
            "target_trade_date",
            "t1_total_return_close",
            "source_exit_fillable",
            "up_limit",
            "down_limit",
        ],
    ].copy()
    for column in IDENTITY_COLUMNS:
        source[column] = source[column].astype(str)
    if source.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise ValueError("V37 candidate identities are duplicated")

    normalized = normalize_historical_minutes(minutes)
    grouped = {
        (str(code), str(date)): group.reset_index(drop=True)
        for (code, date), group in normalized.groupby(
            ["ts_code", "trade_date"],
            sort=False,
        )
    }
    empty = normalized.head(0)
    slippage = float(entry_slippage_bps) / 10_000.0
    cost_pct = float(round_trip_cost_bps) / 100.0
    rows: list[dict[str, Any]] = []

    for record in source.to_dict(orient="records"):
        trade_date = str(record["trade_date"])
        signal_slot = str(record["signal_slot"])
        signal_time = _timestamp(trade_date, signal_slot)
        publication_time = signal_time + pd.offsets.Minute(
            PUBLICATION_DELAY_MINUTES
        )
        entry_time = signal_time + pd.offsets.Minute(ENTRY_DELAY_MINUTES)
        stock_day = grouped.get((str(record["ts_code"]), trade_date), empty)
        entry_bar = stock_day.loc[
            stock_day["trade_time"].eq(entry_time)
        ].tail(1)

        entry_close = _bar_value(entry_bar, "close")
        entry_amount = _bar_value(entry_bar, "amount")
        entry_volume = _bar_value(entry_bar, "vol")
        up_limit = _finite_float(record.get("up_limit"))
        down_limit = _finite_float(record.get("down_limit"))
        up_distance = _distance_to_up_limit(entry_close, up_limit)
        down_distance = _distance_to_down_limit(entry_close, down_limit)
        exact_bar = bool(
            len(entry_bar) == 1
            and not entry_bar["trade_time"].isna().any()
            and entry_bar["trade_time"].iloc[0] == entry_time
        )
        entry_data_complete = bool(
            exact_bar
            and entry_close > 0.0
            and np.isfinite(entry_amount)
            and entry_amount >= 0.0
            and np.isfinite(entry_volume)
            and entry_volume >= 0.0
            and np.isfinite(up_distance)
            and np.isfinite(down_distance)
        )
        entry_fillable = bool(
            entry_data_complete
            and _as_bool(record.get("execution_eligible"))
            and entry_amount >= float(min_slot_amount)
            and (
                entry_amount * float(max_entry_pct_of_slot_amount)
                >= float(reference_order_notional)
            )
            and entry_volume > 0.0
            and up_distance >= float(min_distance_to_up_limit_pct)
            and down_distance >= float(min_distance_to_down_limit_pct)
        )
        exit_fillable = _as_bool(record.get("source_exit_fillable"))
        adjusted_exit = _finite_float(record.get("t1_total_return_close"))
        source_label_available = _as_bool(
            record.get("source_label_available")
        )
        exit_truth_known = bool(
            not entry_fillable or not exit_fillable or adjusted_exit > 0.0
        )
        label_available = bool(
            entry_data_complete
            and source_label_available
            and exit_truth_known
        )
        entry_price = (
            entry_close * (1.0 + slippage)
            if entry_data_complete
            else np.nan
        )
        round_trip_fill = bool(
            label_available and entry_fillable and exit_fillable
        )
        conditional_gross = (
            (adjusted_exit / entry_price - 1.0) * 100.0
            if round_trip_fill
            else np.nan
        )
        conditional_net = (
            conditional_gross - cost_pct
            if np.isfinite(conditional_gross)
            else np.nan
        )
        gross_return = (
            conditional_gross
            if round_trip_fill
            else (0.0 if label_available and not entry_fillable else np.nan)
        )
        net_return = (
            conditional_net
            if round_trip_fill
            else (
                0.0
                if label_available and not entry_fillable
                else (
                    float(non_fill_penalty_pct)
                    if label_available
                    and entry_fillable
                    and not exit_fillable
                    else np.nan
                )
            )
        )
        rows.append(
            {
                **{
                    column: record[column]
                    for column in (*IDENTITY_COLUMNS, "fold")
                },
                "target_trade_date": str(record["target_trade_date"]),
                "v37_publication_time": publication_time.strftime("%H:%M"),
                "v37_entry_benchmark_time": entry_time.strftime("%H:%M"),
                "v37_entry_bar_time": (
                    entry_bar["trade_time"].iloc[0].isoformat()
                    if exact_bar
                    else None
                ),
                "v37_entry_bar_close": (
                    float(entry_close)
                    if np.isfinite(entry_close)
                    else np.nan
                ),
                "v37_entry_bar_amount": (
                    float(entry_amount)
                    if np.isfinite(entry_amount)
                    else np.nan
                ),
                "v37_entry_bar_volume": (
                    float(entry_volume)
                    if np.isfinite(entry_volume)
                    else np.nan
                ),
                "v37_entry_distance_to_up_limit_pct": up_distance,
                "v37_entry_distance_to_down_limit_pct": down_distance,
                "v37_entry_bar_exact": exact_bar,
                "v37_entry_data_complete": entry_data_complete,
                "entry_price": entry_price,
                "gross_return_pct": gross_return,
                "net_return_pct": net_return,
                "entry_fillable": entry_fillable,
                "exit_fillable": exit_fillable,
                "execution_success": round_trip_fill,
                "label_available": label_available,
                "target_net_positive": (
                    int(round_trip_fill and conditional_net > 0.0)
                    if label_available
                    else np.nan
                ),
                "target_severe_loss": (
                    int(np.isfinite(net_return) and net_return <= -2.0)
                    if label_available
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


def join_fast_entry_outcomes(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    identities = list(IDENTITY_COLUMNS)
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        *QUALITY_COLUMNS,
        *OUTCOME_COLUMNS,
    }
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"V37 outcome frame missing columns: {missing}")
    left = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()
    for column in IDENTITY_COLUMNS:
        left[column] = left[column].astype(str)
    left.drop(
        columns=[
            *OUTCOME_COLUMNS,
            "target_trade_date",
            "target_severe_loss",
        ],
        errors="ignore",
        inplace=True,
    )
    right = outcomes.copy()
    for column in IDENTITY_COLUMNS:
        right[column] = right[column].astype(str)
    right.rename(columns={"fold": "_v37_outcome_fold"}, inplace=True)
    result = left.merge(
        right,
        on=identities,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("V37 outcome join missed legal source identities")
    if not pd.to_numeric(result["fold"], errors="coerce").equals(
        pd.to_numeric(result["_v37_outcome_fold"], errors="coerce")
    ):
        raise RuntimeError("V37 outcome folds differ from source folds")
    return result.drop(columns=["_v37_outcome_fold", "_merge"])


def audit_fast_entry_outcomes(
    outcomes: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    legal = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()
    identity = list(IDENTITY_COLUMNS)
    left = (
        legal[identity]
        .astype(str)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    right = (
        outcomes[identity]
        .astype(str)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    identity_exact = bool(
        len(left) == len(right)
        and left.equals(right)
        and not outcomes.duplicated(identity, keep=False).any()
    )
    complete = _boolean(outcomes, "v37_entry_data_complete")
    labels = _boolean(outcomes, "label_available")
    net = _numeric(outcomes, "net_return_pct")
    target = _numeric(outcomes, "target_net_positive")
    consistent = target.eq(net.gt(0.0).astype(float))
    coverage = float(complete.mean()) if len(outcomes) else 0.0
    label_rate = float(labels.mean()) if len(outcomes) else 0.0
    exact_timing = bool(
        _boolean(outcomes, "v37_entry_bar_exact").all()
        and outcomes["v37_entry_benchmark_time"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1423", "1448")
        .all()
    )
    passed = bool(
        identity_exact
        and coverage >= MINIMUM_DATASET_COVERAGE
        and label_rate >= MINIMUM_DATASET_COVERAGE
        and exact_timing
        and consistent.loc[labels].all()
    )
    return {
        "expected_rows": int(len(legal)),
        "outcome_rows": int(len(outcomes)),
        "identity_exact": identity_exact,
        "duplicate_identities": int(outcomes.duplicated(identity).sum()),
        "entry_data_complete_rate": coverage,
        "label_available_rate": label_rate,
        "exact_entry_timing": exact_timing,
        "consistent_labels": bool(consistent.loc[labels].all()),
        "coverage_passed": passed,
    }


def calibrate_fast_entry_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: FastEntryPolicySpec | None = None,
) -> FrozenFastEntryPolicy:
    frozen_spec = spec or FastEntryPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V37 policy calibration has no dates")
    eligible = fast_entry_policy_eligible_rows(scored_calibration)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v34_path_score"]
        .max()
        .sort_values(ascending=False)
    )
    target_days = max(
        1,
        int(np.ceil(frozen_spec.target_candidate_day_rate * len(dates))),
    )
    threshold = (
        float("inf")
        if daily_max.empty
        else float(daily_max.iloc[min(target_days, len(daily_max)) - 1])
    )
    return FrozenFastEntryPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_fast_entry_policy(
    scored: pd.DataFrame,
    policy: FrozenFastEntryPolicy,
) -> pd.DataFrame:
    eligible = fast_entry_policy_eligible_rows(scored)
    qualified = eligible.loc[
        _numeric(eligible, "v34_path_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v37_policy_id"] = policy.policy_id
        return qualified
    qualified["_v37_slot_minute"] = _slot_minutes(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v37_slot_minute",
            "v34_path_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    )
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_v37_slot_minute")
    selected["v37_policy_id"] = policy.policy_id
    selected["v37_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def fast_entry_policy_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    eligible = v34_policy_eligible_rows(scored)
    return eligible.loc[
        eligible["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenFastEntryPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V37 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V37 selected output exceeds fixed daily maximum")
    if not selected["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS).all():
        raise RuntimeError("V37 selected output contains an illegal slot")
    if not _boolean(selected, "v23_point_in_time_complete").all():
        raise RuntimeError("V37 selected output contains incomplete PIT data")
    if not _boolean(selected, "v34_path_complete").all():
        raise RuntimeError("V37 selected output contains incomplete path data")


def selected_execution_audit(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected_rows": 0,
            "entry_fillable_rows": 0,
            "entry_fill_rate": 0.0,
            "round_trip_fill_rows": 0,
            "round_trip_fill_rate": 0.0,
            "exit_fill_rate_given_entry": 0.0,
        }
    entry = _boolean(selected, "entry_fillable")
    success = _boolean(selected, "execution_success")
    exit_fill = _boolean(selected, "exit_fillable")
    entry_rows = int(entry.sum())
    return {
        "selected_rows": int(len(selected)),
        "entry_fillable_rows": entry_rows,
        "entry_fill_rate": float(entry.mean()),
        "round_trip_fill_rows": int(success.sum()),
        "round_trip_fill_rate": float(success.mean()),
        "exit_fill_rate_given_entry": (
            float(exit_fill.loc[entry].mean()) if entry_rows else 0.0
        ),
    }


def v37_research_readiness(
    metrics: dict[str, Any],
    *,
    yearly: list[dict[str, Any]],
    execution_audit: dict[str, Any],
    temporal_integrity: bool,
    source_integrity: bool,
    data_integrity: bool,
) -> dict[str, Any]:
    readiness = v36_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=source_integrity,
        data_integrity=data_integrity,
    )
    gates = dict(readiness["gates"])
    gates["minimum_entry_fill_rate"] = (
        float(execution_audit.get("entry_fill_rate", 0.0)) >= 0.98
    )
    gates["minimum_exit_fill_rate_given_entry"] = (
        float(
            execution_audit.get("exit_fill_rate_given_entry", 0.0)
        )
        >= 0.98
    )
    passed = all(gates.values())
    return {
        **readiness,
        "all_historical_gates_passed": passed,
        "gates": gates,
        "failed_gates": [
            name for name, gate_passed in gates.items() if not gate_passed
        ],
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def _timestamp(trade_date: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} "
        f"{hhmm}:00"
    )


def _bar_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return _finite_float(frame[column].iloc[-1])


def _distance_to_up_limit(price: float, up_limit: float) -> float:
    if not np.isfinite(price) or price <= 0 or not np.isfinite(up_limit):
        return np.nan
    return float((up_limit / price - 1.0) * 100.0)


def _distance_to_down_limit(price: float, down_limit: float) -> float:
    if not np.isfinite(price) or price <= 0 or not np.isfinite(down_limit):
        return np.nan
    return float((price / down_limit - 1.0) * 100.0)


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _slot_minutes(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.replace(":", "", regex=False)
    hours = pd.to_numeric(cleaned.str[:2], errors="coerce")
    minutes = pd.to_numeric(cleaned.str[2:4], errors="coerce")
    return hours * 60.0 + minutes


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )

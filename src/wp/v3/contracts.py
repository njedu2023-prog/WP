from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


CN_TZ = ZoneInfo("Asia/Shanghai")
LEGACY_SIGNAL_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
    "14:50",
)
DEFAULT_SIGNAL_SLOTS = (
    "14:00",
)


@dataclass(frozen=True)
class StrategyContract:
    strategy_id: str = "wp_t1_net_profit_v41_1400"
    model_family: str = "v41_1400_dual_cohort"
    timezone: str = "Asia/Shanghai"
    signal_slots: tuple[str, ...] = DEFAULT_SIGNAL_SLOTS
    candidate_freeze_time: str = "14:10"
    clear_live_display_time: str = "15:00"
    exit_contract: str = "T+1_close"
    board_scope: str = "main_board"
    observation_count: int = 5


@dataclass(frozen=True)
class ExecutionContract:
    entry_price_contract: str = "next_5m_close_plus_slippage"
    entry_delay_minutes: int = 5
    entry_execution_deadline: str = "14:05"
    entry_slippage_bps: float = 10.0
    round_trip_cost_bps: float = 25.0
    exit_order_contract: str = "T+1_14:57_down_limit_sell_for_close_auction"
    stress_cost_bps: tuple[float, ...] = (35.0, 50.0)
    min_listing_days: int = 60
    min_price: float = 2.0
    max_price: float = 150.0
    min_prev_20d_amount: float = 50_000_000.0
    min_slot_amount: float = 3_000_000.0
    reference_order_notional: float = 100_000.0
    max_entry_pct_of_slot_amount: float = 0.01
    min_distance_to_up_limit_pct: float = 0.50
    min_distance_to_down_limit_pct: float = 1.00
    max_market_data_age_seconds: int = 420
    min_intraday_snapshot_count: int = 5
    non_fill_penalty_pct: float = -10.0

    @property
    def baseline_all_in_cost_bps(self) -> float:
        return self.entry_slippage_bps + self.round_trip_cost_bps


@dataclass(frozen=True)
class ModelContract:
    policy_implementation_version: str = "wp_v9_nested_oos_policy_1"
    feature_version: str = "wp_v9_causal_features_1"
    minimum_train_days: int = 252
    calibration_days: int = 42
    policy_design_days: int = 84
    policy_confirmation_days: int = 42
    test_days: int = 42
    purge_days: int = 2
    ensemble_windows_days: tuple[int, ...] = (252, 504)
    temporal_half_life_days: int = 252
    cross_section_top_fraction: float = 0.20
    severe_loss_threshold_pct: float = -2.00
    entry_fill_probability_grid: tuple[float, ...] = (0.97, 0.985)
    exit_fill_probability_grid: tuple[float, ...] = (0.985, 0.995)
    probability_grid: tuple[float, ...] = (0.46, 0.50, 0.54)
    conditional_probability_grid: tuple[float, ...] = (0.50, 0.54, 0.58)
    severe_loss_probability_grid: tuple[float, ...] = (0.25, 0.35)
    selection_rank_grid: tuple[float, ...] = (0.97, 0.98, 0.99)
    expected_utility_grid_pct: tuple[float, ...] = (0.00, 0.10, 0.20)
    downside_grid_pct: tuple[float, ...] = (-3.00, -2.00)
    policy_min_design_events: int = 150
    policy_min_design_days: int = 30
    policy_min_confirmation_events: int = 60
    policy_min_confirmation_days: int = 15
    policy_min_win_rate: float = 0.54
    policy_min_wilson_lower: float = 0.50
    policy_min_clustered_lower: float = 0.48
    policy_min_mean_net_return_pct: float = 0.15
    policy_min_profit_factor: float = 1.10
    max_probability_model_spread: float = 0.15
    max_fill_probability_model_spread: float = 0.10
    max_selection_rank_spread: float = 0.25
    max_expected_return_model_spread_pct: float = 1.00
    min_train_rows: int = 20_000
    max_training_rows_per_slot: int = 240
    random_seed: int = 20_260_727


@dataclass(frozen=True)
class PromotionContract:
    mode: str = "shadow"
    minimum_shadow_trading_days: int = 150
    minimum_shadow_candidate_days: int = 50
    minimum_shadow_candidates: int = 250
    minimum_oos_candidates: int = 250
    minimum_oos_win_rate: float = 0.55
    minimum_oos_win_rate_lower: float = 0.52
    minimum_clustered_win_rate_lower: float = 0.52
    minimum_mean_net_return_pct: float = 0.20
    minimum_clustered_mean_return_lower_pct: float = 0.00
    minimum_median_net_return_pct: float = 0.00
    minimum_profit_factor: float = 1.20
    minimum_entry_fill_rate: float = 0.98
    minimum_exit_fill_rate: float = 0.98
    maximum_ece: float = 0.05
    require_50bps_stress_nonnegative: bool = True
    auto_promote_when_all_gates_pass: bool = True


@dataclass(frozen=True)
class HistoryContract:
    start_date: str = "20210726"
    end_date: str = "20260731"
    evaluation_start_date: str = "20230727"
    evaluation_end_date: str = "20260731"
    partition: str = "month"
    tushare_page_size: int = 8_000
    tushare_requests_per_minute: int = 180
    minute_fetch_workers: int = 4
    raw_cache_days: int = 7
    minimum_minute_universe_coverage: float = 0.90


@dataclass(frozen=True)
class EvidenceContract:
    retrospective_start_date: str = "20260501"
    retrospective_end_date: str = "20260731"
    live_shadow_start_date: str = "20260805"
    keep_cohort_statistics_separate: bool = True


@dataclass(frozen=True)
class V3Config:
    strategy: StrategyContract = field(default_factory=StrategyContract)
    execution: ExecutionContract = field(default_factory=ExecutionContract)
    model: ModelContract = field(default_factory=ModelContract)
    promotion: PromotionContract = field(default_factory=PromotionContract)
    history: HistoryContract = field(default_factory=HistoryContract)
    evidence: EvidenceContract = field(default_factory=EvidenceContract)


def _coerce(cls: type[Any], raw: dict[str, Any]) -> Any:
    data = dict(raw or {})
    for key, value in list(data.items()):
        if key in {
            "signal_slots",
            "stress_cost_bps",
            "ensemble_windows_days",
            "entry_fill_probability_grid",
            "exit_fill_probability_grid",
            "probability_grid",
            "conditional_probability_grid",
            "severe_loss_probability_grid",
            "selection_rank_grid",
            "expected_utility_grid_pct",
            "downside_grid_pct",
        }:
            data[key] = tuple(value)
    return cls(**data)


def load_v3_config(path: str | Path = "config/wp_v3.yml") -> V3Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = V3Config(
        strategy=_coerce(StrategyContract, raw.get("strategy", {})),
        execution=_coerce(ExecutionContract, raw.get("execution", {})),
        model=_coerce(ModelContract, raw.get("model", {})),
        promotion=_coerce(PromotionContract, raw.get("promotion", {})),
        history=_coerce(HistoryContract, raw.get("history", {})),
        evidence=_coerce(EvidenceContract, raw.get("evidence", {})),
    )
    validate_contract(config)
    return config


def validate_contract(config: V3Config) -> None:
    if not config.strategy.signal_slots:
        raise ValueError("signal_slots cannot be empty")
    if tuple(sorted(config.strategy.signal_slots)) != config.strategy.signal_slots:
        raise ValueError("signal_slots must be strictly chronological")
    if config.strategy.signal_slots != DEFAULT_SIGNAL_SLOTS:
        raise ValueError("WP has one immutable signal slot: 14:00")
    if config.strategy.candidate_freeze_time <= config.strategy.signal_slots[-1]:
        raise ValueError("candidate freeze must occur after the final signal slot")
    if config.strategy.clear_live_display_time < config.strategy.candidate_freeze_time:
        raise ValueError("live display cannot clear before the candidate ledger freezes")
    if config.strategy.exit_contract != "T+1_close":
        raise ValueError("WP has one immutable exit contract: T+1_close")
    if config.execution.entry_price_contract != "next_5m_close_plus_slippage":
        raise ValueError("WP entry truth must use the next five-minute close")
    if config.execution.entry_delay_minutes != 5:
        raise ValueError("WP entry benchmark delay is fixed at five minutes")
    expected_deadline = max(
        entry_benchmark_slot(slot, config)
        for slot in config.strategy.signal_slots
    )
    if config.execution.entry_execution_deadline != expected_deadline:
        raise ValueError(
            "WP entry execution deadline must equal the final exact "
            "five-minute benchmark slot"
        )
    if any(
        entry_benchmark_slot(slot, config) > config.execution.entry_execution_deadline
        for slot in config.strategy.signal_slots
    ):
        raise ValueError("a signal cannot settle after the entry execution deadline")
    if config.execution.exit_order_contract != (
        "T+1_14:57_down_limit_sell_for_close_auction"
    ):
        raise ValueError("WP exit order contract must participate in the T+1 close")
    if config.promotion.minimum_shadow_trading_days < 150:
        raise ValueError("production promotion requires at least 150 shadow trading days")
    if config.model.policy_design_days < 40:
        raise ValueError("policy design requires at least 40 trading days")
    if config.model.policy_confirmation_days < 20:
        raise ValueError("policy confirmation requires at least 20 trading days")
    if not 0.05 <= config.model.cross_section_top_fraction <= 0.50:
        raise ValueError("cross_section_top_fraction must be in [0.05, 0.50]")
    if any(
        not 0.30 <= value < 1.0
        for value in (
            *config.model.probability_grid,
            *config.model.conditional_probability_grid,
        )
    ):
        raise ValueError("profit probability grids must remain inside [0.30, 1.0)")
    if any(
        not 0.90 <= value < 1.0
        for value in (
            *config.model.entry_fill_probability_grid,
            *config.model.exit_fill_probability_grid,
        )
    ):
        raise ValueError("fill probability grids must remain inside [0.90, 1.0)")
    if any(
        not 0.90 <= value < 1.0
        for value in config.model.selection_rank_grid
    ):
        raise ValueError("selection rank grid must remain inside [0.90, 1.0)")
    if len(set(config.model.ensemble_windows_days)) < 2:
        raise ValueError("temporal ensemble requires at least two distinct windows")
    if config.model.max_training_rows_per_slot < 100:
        raise ValueError("training sample must retain at least 100 rows per slot-day")
    if config.model.max_expected_return_model_spread_pct <= 0:
        raise ValueError("expected-return model spread limit must be positive")
    if config.execution.round_trip_cost_bps <= 0:
        raise ValueError("round-trip cost must be positive")
    if any(
        cost < config.execution.baseline_all_in_cost_bps
        for cost in config.execution.stress_cost_bps
    ):
        raise ValueError("stress costs cannot be below the baseline all-in cost")
    if config.execution.non_fill_penalty_pct >= 0:
        raise ValueError("non-fill penalty must be negative")
    if not 1 <= config.strategy.observation_count <= 20:
        raise ValueError("observation_count must be between 1 and 20")
    if not 0.0 < config.promotion.minimum_entry_fill_rate <= 1.0:
        raise ValueError("minimum_entry_fill_rate must be in (0, 1]")
    if not 0.0 < config.promotion.minimum_exit_fill_rate <= 1.0:
        raise ValueError("minimum_exit_fill_rate must be in (0, 1]")
    if config.history.tushare_requests_per_minute < 30:
        raise ValueError("tushare_requests_per_minute cannot be below 30")
    if not 1 <= config.history.minute_fetch_workers <= 8:
        raise ValueError("minute_fetch_workers must be between 1 and 8")
    history_dates = {
        name: datetime.strptime(value, "%Y%m%d")
        for name, value in (
            ("start_date", config.history.start_date),
            ("end_date", config.history.end_date),
            ("evaluation_start_date", config.history.evaluation_start_date),
            ("evaluation_end_date", config.history.evaluation_end_date),
        )
    }
    if not (
        history_dates["start_date"]
        < history_dates["evaluation_start_date"]
        <= history_dates["evaluation_end_date"]
        <= history_dates["end_date"]
    ):
        raise ValueError(
            "history dates must satisfy start_date < evaluation_start_date "
            "<= evaluation_end_date <= end_date"
        )
    evidence_dates = {
        name: datetime.strptime(value, "%Y%m%d")
        for name, value in (
            (
                "retrospective_start_date",
                config.evidence.retrospective_start_date,
            ),
            (
                "retrospective_end_date",
                config.evidence.retrospective_end_date,
            ),
            ("live_shadow_start_date", config.evidence.live_shadow_start_date),
        )
    }
    if not (
        evidence_dates["retrospective_start_date"]
        <= evidence_dates["retrospective_end_date"]
        < evidence_dates["live_shadow_start_date"]
    ):
        raise ValueError(
            "retrospective evidence must end before live shadow statistics start"
        )
    if not config.evidence.keep_cohort_statistics_separate:
        raise ValueError("qualified and observation statistics must remain separate")


def policy_fingerprint(config: V3Config) -> str:
    payload = {
        "strategy": asdict(config.strategy),
        "execution": asdict(config.execution),
        "model": asdict(config.model),
        "data_quality": {
            "minimum_minute_universe_coverage": (
                config.history.minimum_minute_universe_coverage
            )
        },
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def session_phase(now: datetime | None = None, config: V3Config | None = None) -> str:
    cfg = config or V3Config()
    current = now or datetime.now(CN_TZ)
    hhmm = current.strftime("%H:%M")
    if hhmm < cfg.strategy.signal_slots[0]:
        return "PRE_SIGNAL"
    if hhmm <= cfg.strategy.signal_slots[-1]:
        return "SIGNAL"
    if hhmm < cfg.strategy.candidate_freeze_time:
        return "NO_NEW_SIGNAL"
    if hhmm < cfg.strategy.clear_live_display_time:
        return "FROZEN"
    return "CLOSED"


def due_signal_slot(now: datetime, config: V3Config) -> str | None:
    hhmm = now.astimezone(CN_TZ).strftime("%H:%M")
    due = [slot for slot in config.strategy.signal_slots if slot <= hhmm]
    return due[-1] if due else None


def entry_benchmark_slot(signal_slot: str, config: V3Config) -> str:
    parsed = datetime.strptime(signal_slot, "%H:%M")
    shifted = parsed + timedelta(minutes=config.execution.entry_delay_minutes)
    return shifted.strftime("%H:%M")


def parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()

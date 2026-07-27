from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


CN_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_SIGNAL_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
    "14:50",
)


@dataclass(frozen=True)
class StrategyContract:
    strategy_id: str = "wp_t1_net_profit_v3"
    model_family: str = "calibrated_temporal_ensemble_v3"
    timezone: str = "Asia/Shanghai"
    signal_slots: tuple[str, ...] = DEFAULT_SIGNAL_SLOTS
    candidate_freeze_time: str = "14:55"
    clear_live_display_time: str = "15:00"
    exit_contract: str = "T+1_close"
    board_scope: str = "main_board"


@dataclass(frozen=True)
class ExecutionContract:
    entry_price_contract: str = "signal_price_plus_slippage"
    entry_slippage_bps: float = 10.0
    round_trip_cost_bps: float = 25.0
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
    policy_implementation_version: str = "wp_v3_policy_20260727_9"
    feature_version: str = "wp_v3_causal_features_5"
    minimum_train_days: int = 252
    calibration_days: int = 21
    test_days: int = 42
    purge_days: int = 2
    ensemble_windows_days: tuple[int, ...] = (126, 252, 504)
    probability_threshold: float = 0.60
    probability_lower_threshold: float = 0.52
    min_expected_net_return_pct: float = 0.30
    min_downside_q10_pct: float = -4.50
    min_calibration_bin_samples: int = 100
    min_calibration_bin_days: int = 10
    min_calibration_bin_wilson_lower: float = 0.52
    min_calibration_bin_clustered_lower: float = 0.50
    max_probability_model_spread: float = 0.10
    min_train_rows: int = 20_000
    max_training_rows_per_slot: int = 300
    random_seed: int = 20_260_727


@dataclass(frozen=True)
class PromotionContract:
    mode: str = "shadow"
    minimum_shadow_trading_days: int = 150
    minimum_shadow_candidate_days: int = 80
    minimum_shadow_candidates: int = 250
    minimum_oos_candidates: int = 250
    minimum_oos_win_rate: float = 0.60
    minimum_oos_win_rate_lower: float = 0.52
    minimum_clustered_win_rate_lower: float = 0.52
    minimum_mean_net_return_pct: float = 0.30
    minimum_clustered_mean_return_lower_pct: float = 0.00
    minimum_median_net_return_pct: float = 0.00
    minimum_profit_factor: float = 1.30
    maximum_ece: float = 0.05
    require_50bps_stress_nonnegative: bool = True
    auto_promote_when_all_gates_pass: bool = True


@dataclass(frozen=True)
class HistoryContract:
    start_date: str = "20230727"
    end_date: str = "20260724"
    partition: str = "month"
    tushare_page_size: int = 8_000
    tushare_requests_per_minute: int = 180
    minute_fetch_workers: int = 4
    raw_cache_days: int = 7
    minimum_minute_universe_coverage: float = 0.90


@dataclass(frozen=True)
class V3Config:
    strategy: StrategyContract = field(default_factory=StrategyContract)
    execution: ExecutionContract = field(default_factory=ExecutionContract)
    model: ModelContract = field(default_factory=ModelContract)
    promotion: PromotionContract = field(default_factory=PromotionContract)
    history: HistoryContract = field(default_factory=HistoryContract)


def _coerce(cls: type[Any], raw: dict[str, Any]) -> Any:
    data = dict(raw or {})
    for key, value in list(data.items()):
        if key in {"signal_slots", "stress_cost_bps", "ensemble_windows_days"}:
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
    )
    validate_contract(config)
    return config


def validate_contract(config: V3Config) -> None:
    if tuple(sorted(config.strategy.signal_slots)) != config.strategy.signal_slots:
        raise ValueError("signal_slots must be strictly chronological")
    if any(slot < "14:20" or slot > "14:50" for slot in config.strategy.signal_slots):
        raise ValueError("all signal slots must remain inside 14:20-14:50")
    if config.strategy.candidate_freeze_time <= config.strategy.signal_slots[-1]:
        raise ValueError("candidate freeze must occur after the final signal slot")
    if config.strategy.clear_live_display_time < config.strategy.candidate_freeze_time:
        raise ValueError("live display cannot clear before the candidate ledger freezes")
    if config.strategy.exit_contract != "T+1_close":
        raise ValueError("V3 has one immutable exit contract: T+1_close")
    if config.promotion.minimum_shadow_trading_days < 150:
        raise ValueError("production promotion requires at least 150 shadow trading days")
    if not 0.5 <= config.model.probability_threshold < 1.0:
        raise ValueError("probability_threshold must be in [0.5, 1.0)")
    if len(set(config.model.ensemble_windows_days)) < 2:
        raise ValueError("temporal ensemble requires at least two distinct windows")
    if config.model.max_training_rows_per_slot < 100:
        raise ValueError("training sample must retain at least 100 rows per slot-day")
    if config.execution.round_trip_cost_bps <= 0:
        raise ValueError("round-trip cost must be positive")
    if any(
        cost < config.execution.baseline_all_in_cost_bps
        for cost in config.execution.stress_cost_bps
    ):
        raise ValueError("stress costs cannot be below the baseline all-in cost")
    if config.execution.non_fill_penalty_pct >= 0:
        raise ValueError("non-fill penalty must be negative")
    if config.history.tushare_requests_per_minute < 30:
        raise ValueError("tushare_requests_per_minute cannot be below 30")
    if not 1 <= config.history.minute_fetch_workers <= 8:
        raise ValueError("minute_fetch_workers must be between 1 and 8")


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


def parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()

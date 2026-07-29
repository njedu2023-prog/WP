from __future__ import annotations

from wp.v3.contracts import V3Config
from wp.v3.registry import (
    empty_registry,
    evaluate_promotion,
    refresh_shadow_metrics,
    register_research_model,
)


def test_model_cannot_promote_before_150_shadow_trading_days():
    record = {
        "backtest": {"backtest_gate": {"passed": True}},
        "shadow": {
            "evidence_scope": "exact_model",
            "model_fingerprint": "model-a",
            "trading_days": 149,
            "candidate_days": 100,
            "verified_candidates": 300,
            "win_rate": 0.66,
            "win_rate_wilson_lower": 0.58,
            "mean_net_return_pct": 0.8,
            "median_net_return_pct": 0.3,
            "profit_factor": 1.8,
            "entry_fill_rate": 0.99,
            "exit_fill_rate": 0.99,
            "ece": 0.03,
            "stress_50bps_positive_total_return": True,
        },
    }
    record["fingerprint"] = "model-a"
    decision = evaluate_promotion(record, V3Config())
    assert decision.eligible is False
    assert decision.checks["shadow_trading_days"] is False


def test_shadow_days_only_count_sessions_with_the_same_model_fingerprint():
    registry = {
        "schema_version": "wp_model_registry_v3",
        "active_model_fingerprint": None,
        "shadow_model_fingerprint": "abc",
        "models": [
            {
                "fingerprint": "abc",
                "status": "SHADOW",
                "trained_at": "2026-07-23T00:00:00Z",
                "train_end": "20260722",
                "backtest": {"backtest_gate": {"passed": False}},
                "shadow": {},
            }
        ],
    }
    ledger = {
        "schema_version": "wp_candidate_ledger_v3",
        "sessions": [
            {
                "trade_date": "20260722",
                "frozen": True,
                "frozen_at": "2026-07-22T14:55:00+08:00",
                "model_fingerprint": None,
                "candidates": [],
            },
            {
                "trade_date": "20260723",
                "frozen": True,
                "frozen_at": "2026-07-23T14:55:00+08:00",
                "model_fingerprint": "different",
                "candidates": [],
            },
            {
                "trade_date": "20260724",
                "frozen": True,
                "frozen_at": "2026-07-24T14:55:00+08:00",
                "model_fingerprint": "abc",
                "candidates": [],
            },
        ],
    }
    refresh_shadow_metrics(registry, "abc", ledger, V3Config())
    assert registry["models"][0]["shadow"]["trading_days"] == 1
    assert registry["models"][0]["shadow"]["started_trade_date"] == "20260724"


def test_historical_backfill_cannot_count_as_forward_shadow_day():
    registry = empty_registry()
    register_research_model(
        registry,
        metadata=_metadata(
            "model-a",
            "policy-a",
            "2026-07-27T00:00:00Z",
            train_end="20260724",
        ),
        backtest={"backtest_gate": {"passed": True}},
        artifact_path="a.joblib",
    )
    ledger = {
        "schema_version": "wp_candidate_ledger_v3",
        "sessions": [
            {
                "trade_date": "20260727",
                "frozen": True,
                "frozen_at": "2026-07-28T10:00:00+08:00",
                "model_fingerprint": "model-a",
                "policy_fingerprint": "policy-a",
                "candidates": [],
            },
            {
                "trade_date": "20260728",
                "frozen": True,
                "frozen_at": "2026-07-28T14:55:00+08:00",
                "model_fingerprint": "model-a",
                "policy_fingerprint": "policy-a",
                "candidates": [],
            },
        ],
    }

    refresh_shadow_metrics(registry, "model-a", ledger, V3Config())

    shadow = registry["models"][0]["shadow"]
    assert shadow["trading_days"] == 1
    assert shadow["started_trade_date"] == "20260728"
    assert shadow["observation_after_trade_date"] == "20260724"


def test_shadow_cost_stress_only_charges_filled_entries():
    registry = empty_registry()
    register_research_model(
        registry,
        metadata=_metadata(
            "model-a",
            "policy-a",
            "2026-07-27T00:00:00Z",
            train_end="20260724",
        ),
        backtest={"backtest_gate": {"passed": True}},
        artifact_path="a.joblib",
    )
    ledger = {
        "schema_version": "wp_candidate_ledger_v3",
        "sessions": [
            {
                "trade_date": "20260728",
                "frozen": True,
                "frozen_at": "2026-07-28T14:55:00+08:00",
                "model_fingerprint": "model-a",
                "policy_fingerprint": "policy-a",
                "candidates": [
                    {
                        "trade_date": "20260728",
                        "model_fingerprint": "model-a",
                        "truth_status": "verified",
                        "net_return_pct": 0.0,
                        "p_net_positive": 0.2,
                        "entry_fillable": False,
                        "exit_fillable": False,
                    },
                    {
                        "trade_date": "20260728",
                        "model_fingerprint": "model-a",
                        "truth_status": "verified",
                        "net_return_pct": 0.2,
                        "p_net_positive": 0.6,
                        "entry_fillable": True,
                        "exit_fillable": True,
                    },
                ],
            }
        ],
    }

    refresh_shadow_metrics(registry, "model-a", ledger, V3Config())

    shadow = registry["models"][0]["shadow"]
    assert shadow["stress_50bps_positive_total_return"] is True


def _metadata(
    fingerprint: str,
    policy: str,
    trained_at: str,
    *,
    train_end: str = "20260701",
) -> dict:
    return {
        "fingerprint": fingerprint,
        "policy_fingerprint": policy,
        "training_data_digest": f"digest-{fingerprint}",
        "model_version": fingerprint,
        "feature_version": "features",
        "trained_at": trained_at,
        "train_start": "20230101",
        "train_end": train_end,
    }


def test_same_policy_monthly_retrain_cannot_inherit_shadow_clock():
    registry = empty_registry()
    backtest = {"backtest_gate": {"passed": True}}
    register_research_model(
        registry,
        metadata=_metadata(
            "model-a",
            "policy-a",
            "2026-06-01T00:00:00Z",
            train_end="20260529",
        ),
        backtest=backtest,
        artifact_path="a.joblib",
    )
    register_research_model(
        registry,
        metadata=_metadata(
            "model-b",
            "policy-a",
            "2026-07-01T00:00:00Z",
            train_end="20260630",
        ),
        backtest=backtest,
        artifact_path="b.joblib",
    )
    assert registry["shadow_model_fingerprint"] == "model-a"
    assert registry["shadow_policy_fingerprint"] == "policy-a"
    assert registry["models"][0]["status"] == "SHADOW"
    assert registry["models"][1]["status"] == "RESEARCH"

    ledger = {
        "schema_version": "wp_candidate_ledger_v3",
        "sessions": [
            {
                "trade_date": "20260630",
                "frozen": True,
                "frozen_at": "2026-06-30T14:55:00+08:00",
                "model_fingerprint": "model-a",
                "policy_fingerprint": "policy-a",
                "candidates": [],
            },
            {
                "trade_date": "20260701",
                "frozen": True,
                "frozen_at": "2026-07-01T14:55:00+08:00",
                "model_fingerprint": "model-b",
                "policy_fingerprint": "policy-a",
                "candidates": [],
            },
        ],
    }
    refresh_shadow_metrics(registry, "model-a", ledger, V3Config())
    assert registry["models"][0]["shadow"]["trading_days"] == 1
    assert registry["models"][1]["shadow"]["trading_days"] == 0


def test_material_policy_change_does_not_inherit_shadow_evidence():
    registry = empty_registry()
    passed = {"backtest_gate": {"passed": True}}
    failed = {"backtest_gate": {"passed": False}}
    register_research_model(
        registry,
        metadata=_metadata("model-a", "policy-a", "2026-06-01T00:00:00Z"),
        backtest=passed,
        artifact_path="a.joblib",
    )
    register_research_model(
        registry,
        metadata=_metadata("model-b", "policy-b", "2026-07-01T00:00:00Z"),
        backtest=failed,
        artifact_path="b.joblib",
    )
    assert registry["models"][1]["status"] == "RESEARCH"
    assert registry["shadow_policy_fingerprint"] == "policy-a"


def test_backtest_failed_policy_starts_observation_but_cannot_promote():
    registry = empty_registry()
    register_research_model(
        registry,
        metadata=_metadata("model-a", "policy-a", "2026-06-01T00:00:00Z"),
        backtest={"backtest_gate": {"passed": False}},
        artifact_path="a.joblib",
    )
    assert registry["models"][0]["status"] == "SHADOW_OBSERVATION"
    assert registry["shadow_model_fingerprint"] == "model-a"
    assert registry["shadow_policy_fingerprint"] == "policy-a"
    assert evaluate_promotion(registry["models"][0], V3Config()).eligible is False


def test_failed_challenger_does_not_replace_backtest_passed_shadow():
    registry = empty_registry()
    register_research_model(
        registry,
        metadata=_metadata("model-a", "policy-a", "2026-06-01T00:00:00Z"),
        backtest={"backtest_gate": {"passed": True}},
        artifact_path="a.joblib",
    )
    register_research_model(
        registry,
        metadata=_metadata("model-b", "policy-b", "2026-07-01T00:00:00Z"),
        backtest={"backtest_gate": {"passed": False}},
        artifact_path="b.joblib",
    )

    assert registry["shadow_model_fingerprint"] == "model-a"
    assert registry["models"][1]["status"] == "RESEARCH"

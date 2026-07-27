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
            "trading_days": 149,
            "candidate_days": 100,
            "verified_candidates": 300,
            "win_rate": 0.66,
            "win_rate_wilson_lower": 0.58,
            "mean_net_return_pct": 0.8,
            "median_net_return_pct": 0.3,
            "profit_factor": 1.8,
            "ece": 0.03,
            "stress_50bps_positive_total_return": True,
        },
    }
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
                "model_fingerprint": None,
                "candidates": [],
            },
            {
                "trade_date": "20260723",
                "frozen": True,
                "model_fingerprint": "different",
                "candidates": [],
            },
            {
                "trade_date": "20260724",
                "frozen": True,
                "model_fingerprint": "abc",
                "candidates": [],
            },
        ],
    }
    refresh_shadow_metrics(registry, "abc", ledger, V3Config())
    assert registry["models"][0]["shadow"]["trading_days"] == 1
    assert registry["models"][0]["shadow"]["started_trade_date"] == "20260724"


def _metadata(fingerprint: str, policy: str, trained_at: str) -> dict:
    return {
        "fingerprint": fingerprint,
        "policy_fingerprint": policy,
        "training_data_digest": f"digest-{fingerprint}",
        "model_version": fingerprint,
        "feature_version": "features",
        "trained_at": trained_at,
        "train_start": "20230101",
        "train_end": "20260701",
    }


def test_same_policy_monthly_retrain_keeps_one_shadow_clock():
    registry = empty_registry()
    backtest = {"backtest_gate": {"passed": False}}
    register_research_model(
        registry,
        metadata=_metadata("model-a", "policy-a", "2026-06-01T00:00:00Z"),
        backtest=backtest,
        artifact_path="a.joblib",
    )
    register_research_model(
        registry,
        metadata=_metadata("model-b", "policy-a", "2026-07-01T00:00:00Z"),
        backtest=backtest,
        artifact_path="b.joblib",
    )
    assert registry["shadow_model_fingerprint"] == "model-b"
    assert registry["shadow_policy_fingerprint"] == "policy-a"
    assert registry["models"][0]["status"] == "SUPERSEDED_SHADOW"
    assert registry["models"][1]["status"] == "SHADOW"

    ledger = {
        "schema_version": "wp_candidate_ledger_v3",
        "sessions": [
            {
                "trade_date": "20260630",
                "frozen": True,
                "model_fingerprint": "model-a",
                "policy_fingerprint": "policy-a",
                "candidates": [],
            },
            {
                "trade_date": "20260701",
                "frozen": True,
                "model_fingerprint": "model-b",
                "policy_fingerprint": "policy-a",
                "candidates": [],
            },
        ],
    }
    refresh_shadow_metrics(registry, "model-b", ledger, V3Config())
    assert registry["models"][1]["shadow"]["trading_days"] == 2


def test_material_policy_change_does_not_inherit_shadow_evidence():
    registry = empty_registry()
    backtest = {"backtest_gate": {"passed": False}}
    register_research_model(
        registry,
        metadata=_metadata("model-a", "policy-a", "2026-06-01T00:00:00Z"),
        backtest=backtest,
        artifact_path="a.joblib",
    )
    register_research_model(
        registry,
        metadata=_metadata("model-b", "policy-b", "2026-07-01T00:00:00Z"),
        backtest=backtest,
        artifact_path="b.joblib",
    )
    assert registry["models"][1]["status"] == "RESEARCH"
    assert registry["shadow_policy_fingerprint"] == "policy-a"

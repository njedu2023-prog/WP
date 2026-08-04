from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.research_wp_v10_meta import (
    META_CALIBRATION_DAYS,
    META_TRAIN_DAYS,
    POLICY_CONFIRMATION_DAYS,
    POLICY_DESIGN_DAYS,
    PURGE_DAYS,
    _json_default,
    _rolling_segments,
)
from wp.v3.meta_alpha import (
    META_FEATURE_COLUMNS,
    MetaPolicy,
    ProbabilityCalibrator,
    apply_meta_policy,
    attach_meta_context,
    fit_meta_alpha,
    prune_candidate_universe,
)


def test_context_features_are_causal_and_pruning_is_deterministic() -> None:
    rows = []
    for code_index in range(1, 8):
        rows.append(
            {
                "trade_date": "20260721",
                "signal_slot": "14:20",
                "ts_code": f"00000{code_index}.SZ",
                "execution_eligible": True,
                "label_available": True,
                "ret_from_prev_close_pct": float(code_index),
                "p_net_positive": code_index / 10.0,
                "p_net_positive_lower": code_index / 12.0,
                "p_conditional_net_positive": code_index / 11.0,
                "p_severe_loss": (8 - code_index) / 10.0,
                "selection_score": float(code_index),
                "selection_rank_pct": code_index / 7.0,
                "expected_utility_pct": float(code_index) / 5.0,
                "expected_utility_lower_pct": float(code_index) / 7.0,
                "p_entry_fill": 0.95,
                "p_exit_fill_given_entry": 0.98,
                "p_round_trip_fill_lower": 0.90,
                "downside_q10_pct": -2.0,
                "target_net_positive": code_index % 2,
                "target_severe_loss": code_index % 3 == 0,
                "net_return_pct": float(code_index - 4),
            }
        )
    frame = pd.DataFrame(rows)
    contextual = attach_meta_context(frame)
    changed_truth = frame.copy()
    changed_truth["target_net_positive"] = 1
    changed_truth["target_severe_loss"] = 0
    changed_truth["net_return_pct"] = 999.0
    contextual_changed = attach_meta_context(changed_truth)
    pd.testing.assert_frame_equal(
        contextual.loc[:, META_FEATURE_COLUMNS],
        contextual_changed.loc[:, META_FEATURE_COLUMNS],
    )

    first = prune_candidate_universe(frame, top_per_score=2)
    second = prune_candidate_universe(
        frame.sample(frac=1.0, random_state=7),
        top_per_score=2,
    )
    assert first["ts_code"].tolist() == second["ts_code"].tolist()
    assert "000007.SZ" in set(first["ts_code"])


def test_policy_selection_is_chronological_and_locks_first_signal() -> None:
    frame = pd.DataFrame(
        [
            _scored("20260721", "14:20", "000001.SZ", 0.90),
            _scored("20260721", "14:20", "000002.SZ", 0.80),
            _scored("20260721", "14:25", "000001.SZ", 0.99),
            _scored("20260721", "14:25", "000003.SZ", 0.98),
            _scored("20260722", "14:45", "000004.SZ", 0.95),
        ]
    )
    policy = MetaPolicy(
        probability_min=0.50,
        expected_return_min_pct=0.0,
        severe_loss_max=0.35,
        round_trip_fill_min=0.90,
        meta_rank_min=0.50,
        max_candidates_per_day=2,
        slot_group="all",
    )
    selected = apply_meta_policy(frame, policy)
    day = selected.loc[selected["trade_date"].eq("20260721")]
    assert day["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert day["signal_slot"].tolist() == ["14:20", "14:20"]
    assert not selected.duplicated(["trade_date", "ts_code"]).any()


def test_rolling_segments_include_explicit_purges() -> None:
    needed = (
        META_TRAIN_DAYS
        + META_CALIBRATION_DAYS
        + POLICY_DESIGN_DAYS
        + POLICY_CONFIRMATION_DAYS
        + 4 * PURGE_DAYS
    )
    dates = [f"{index:08d}" for index in range(1, needed + 1)]
    train, calibration, design, confirmation = _rolling_segments(dates)  # type: ignore[misc]
    assert len(train) == META_TRAIN_DAYS
    assert len(calibration) == META_CALIBRATION_DAYS
    assert len(design) == POLICY_DESIGN_DAYS
    assert len(confirmation) == POLICY_CONFIRMATION_DAYS
    assert int(calibration[0]) - int(train[-1]) == PURGE_DAYS + 1
    assert int(design[0]) - int(calibration[-1]) == PURGE_DAYS + 1
    assert int(confirmation[0]) - int(design[-1]) == PURGE_DAYS + 1
    assert int(dates[-1]) - int(confirmation[-1]) == PURGE_DAYS


def test_meta_model_produces_finite_oos_scores() -> None:
    rng = np.random.default_rng(20260730)
    train = _model_frame(rng, 1_400, "2025")
    calibration = _model_frame(rng, 500, "2026")
    test = _model_frame(rng, 120, "2027")
    for frame in (train, calibration, test):
        frame["data_age_seconds"] = np.nan
        frame["p_entry_fill"] = 0.95
    bundle = fit_meta_alpha(train, calibration, random_seed=7)
    scored = bundle.predict(test)
    assert "data_age_seconds" not in bundle.feature_columns
    assert "p_entry_fill" not in bundle.feature_columns
    assert scored["meta_p_positive"].between(0.001, 0.999).all()
    assert scored["meta_p_positive_lower"].between(0.001, 0.999).all()
    assert (
        scored["meta_p_positive_lower"] <= scored["meta_p_positive"]
    ).all()
    assert scored["meta_p_positive"].nunique() > 20
    assert scored["meta_p_positive_raw"].nunique() > 20
    assert np.isfinite(scored["meta_probability_calibration_margin"]).all()
    assert scored["meta_p_severe_loss"].between(0.001, 0.999).all()
    assert np.isfinite(scored["meta_expected_net_return_pct"]).all()
    assert np.isfinite(scored["meta_score"]).all()


def test_platt_calibration_preserves_resolution_and_builds_a_lower_bound() -> None:
    raw = np.linspace(0.10, 0.90, 240)
    target = (
        raw
        + 0.12 * np.sin(np.arange(len(raw)) / 7.0)
        > 0.52
    ).astype(int)
    dates = np.asarray(
        [f"2026{index // 20 + 1:02d}{index % 20 + 1:02d}" for index in range(240)]
    )
    weights = np.ones(len(raw), dtype=float)
    calibrator = ProbabilityCalibrator(method="platt").fit(
        raw,
        target,
        weights,
        dates=dates,
        margin_seed=17,
    )

    point = calibrator.predict(raw)
    lower = calibrator.predict_lower(
        raw,
        member_probabilities=np.column_stack(
            [np.clip(raw - 0.04, 0.001, 0.999), raw]
        ),
    )

    assert np.unique(np.round(point, 8)).size > 200
    assert np.all(np.diff(point) >= 0)
    assert (lower <= point).all()
    assert calibrator.one_sided_margin >= 0.0


def test_research_summary_serializes_numpy_scalars() -> None:
    payload = {
        "fold": np.int64(11),
        "mean": np.float64(-0.0677),
        "authorized": np.bool_(False),
    }
    assert json.loads(json.dumps(payload, default=_json_default)) == {
        "fold": 11,
        "mean": -0.0677,
        "authorized": False,
    }


def _scored(
    trade_date: str,
    signal_slot: str,
    ts_code: str,
    score: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": signal_slot,
        "ts_code": ts_code,
        "meta_p_positive": 0.60,
        "meta_expected_net_return_pct": 0.30,
        "meta_p_severe_loss": 0.10,
        "p_round_trip_fill_lower": 0.96,
        "meta_rank_pct": 0.99,
        "meta_score": score,
        "net_return_pct": 1.0,
        "entry_fillable": True,
        "exit_fillable": True,
    }


def _model_frame(
    rng: np.random.Generator,
    rows: int,
    year: str,
) -> pd.DataFrame:
    dates = [
        f"{year}{month:02d}{day:02d}"
        for month in range(1, 13)
        for day in range(1, 21)
    ]
    frame = pd.DataFrame(
        {
            "trade_date": [dates[index % len(dates)] for index in range(rows)],
            "signal_slot": [
                ("14:20", "14:30", "14:45")[index % 3]
                for index in range(rows)
            ],
            "ts_code": [f"{index % 500:06d}.SZ" for index in range(rows)],
            "execution_eligible": True,
            "label_available": True,
            "entry_fillable": rng.random(rows) > 0.06,
            "exit_fillable": rng.random(rows) > 0.03,
            "ret_from_prev_close_pct": rng.normal(2.0, 3.0, rows),
            "p_entry_fill": rng.uniform(0.85, 1.0, rows),
            "p_exit_fill_given_entry": rng.uniform(0.93, 1.0, rows),
            "p_round_trip_fill_lower": rng.uniform(0.82, 0.99, rows),
            "p_net_positive": rng.uniform(0.25, 0.75, rows),
            "p_net_positive_lower": rng.uniform(0.20, 0.65, rows),
            "p_conditional_net_positive": rng.uniform(0.30, 0.80, rows),
            "p_severe_loss": rng.uniform(0.05, 0.45, rows),
            "selection_score": rng.normal(0.0, 1.0, rows),
            "selection_rank_pct": rng.uniform(0.0, 1.0, rows),
            "expected_utility_pct": rng.normal(0.0, 0.8, rows),
            "expected_utility_lower_pct": rng.normal(-0.4, 0.8, rows),
            "downside_q10_pct": rng.normal(-3.0, 1.0, rows),
            "probability_model_spread": rng.uniform(0.0, 0.2, rows),
            "fill_probability_model_spread": rng.uniform(0.0, 0.12, rows),
            "selection_rank_spread": rng.uniform(0.0, 0.3, rows),
            "expected_return_model_spread": rng.uniform(0.0, 1.2, rows),
            "data_age_seconds": rng.uniform(0.0, 420.0, rows),
        }
    )
    context = attach_meta_context(frame)
    latent = (
        0.55 * context["probability_rank_pct"].fillna(0.5)
        + 0.25 * context["return_context_zscore"].fillna(0.0)
        - 0.35 * frame["p_severe_loss"]
        + rng.normal(0.0, 0.25, rows)
    )
    target = latent.gt(latent.median()).astype(int)
    frame["target_net_positive"] = target
    frame["target_severe_loss"] = (
        frame["p_severe_loss"] + rng.normal(0.0, 0.1, rows)
    ).gt(0.35)
    frame["net_return_pct"] = (
        np.where(target.eq(1), 1.2, -1.0)
        + rng.normal(0.0, 0.5, rows)
    )
    return attach_meta_context(frame)

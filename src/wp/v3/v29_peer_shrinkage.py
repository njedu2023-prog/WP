from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wp.v3.v28_industry_peer import (
    MINIMUM_L2_PEERS,
    MINIMUM_L3_PEERS,
    PEER_FEATURE_SUFFIXES,
    V28_PEER_FEATURE_COLUMNS,
)


IDENTITY_COLUMNS = ("trade_date", "signal_slot", "ts_code")
L3_SHRINKAGE_PSEUDO_PEERS = 6.0
SHRUNK_METRIC_SUFFIXES = tuple(
    suffix
    for suffix in PEER_FEATURE_SUFFIXES
    if suffix not in {"count", "return_count"}
)
V29_CONTEXT_FEATURE_COLUMNS = (
    "v29_peer_log_l2_count",
    "v29_peer_log_l3_count",
    "v29_peer_l2_available",
    "v29_peer_l3_available",
    "v29_peer_any_available",
    "v29_peer_no_peer_context",
    "v29_peer_l2_shallow",
    "v29_peer_l3_shallow",
    "v29_peer_l3_weight",
    "v29_peer_l3_share_of_l2",
)
V29_SHRUNK_FEATURE_COLUMNS = tuple(
    f"v29_peer_{suffix}" for suffix in SHRUNK_METRIC_SUFFIXES
)
V29_FEATURE_COLUMNS = (
    *V29_CONTEXT_FEATURE_COLUMNS,
    *V29_SHRUNK_FEATURE_COLUMNS,
)


def build_hierarchical_peer_features(
    v28_features: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        *V28_PEER_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(v28_features.columns))
    if missing:
        raise ValueError(f"V29 source is missing V28 columns: {missing}")
    if v28_features.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("V29 source contains duplicate candidate identities")

    source = v28_features.copy()
    numeric = source.loc[:, V28_PEER_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    l2_count = numeric["v28_l2_peer_count"].fillna(0.0).clip(lower=0.0)
    l3_count = numeric["v28_l3_peer_count"].fillna(0.0).clip(lower=0.0)
    l2_available = l2_count.gt(0.0)
    l3_available = l3_count.gt(0.0)
    any_available = l2_available | l3_available

    result = source.loc[:, [*IDENTITY_COLUMNS, "fold"]].copy()
    result["v29_peer_log_l2_count"] = np.log1p(l2_count)
    result["v29_peer_log_l3_count"] = np.log1p(l3_count)
    result["v29_peer_l2_available"] = l2_available.astype(float)
    result["v29_peer_l3_available"] = l3_available.astype(float)
    result["v29_peer_any_available"] = any_available.astype(float)
    result["v29_peer_no_peer_context"] = (~any_available).astype(float)
    result["v29_peer_l2_shallow"] = l2_count.lt(
        MINIMUM_L2_PEERS
    ).astype(float)
    result["v29_peer_l3_shallow"] = l3_count.lt(
        MINIMUM_L3_PEERS
    ).astype(float)
    raw_weight = l3_count / (
        l3_count + float(L3_SHRINKAGE_PSEUDO_PEERS)
    )
    result["v29_peer_l3_weight"] = raw_weight.where(
        l2_available & l3_available,
        l3_available.astype(float),
    )
    result["v29_peer_l3_share_of_l2"] = (
        l3_count / l2_count.where(l2_count.gt(0.0), np.nan)
    ).clip(lower=0.0, upper=1.0).fillna(0.0)

    for suffix in SHRUNK_METRIC_SUFFIXES:
        l2 = numeric[f"v28_l2_peer_{suffix}"]
        l3 = numeric[f"v28_l3_peer_{suffix}"]
        result[f"v29_peer_{suffix}"] = shrink_metric(
            l2,
            l3,
            l3_count=l3_count,
        )

    result.loc[:, V29_FEATURE_COLUMNS] = result.loc[
        :, V29_FEATURE_COLUMNS
    ].replace([np.inf, -np.inf], np.nan)
    if result.loc[:, V29_FEATURE_COLUMNS].isna().any().any():
        missing_counts = {
            column: int(result[column].isna().sum())
            for column in V29_FEATURE_COLUMNS
            if result[column].isna().any()
        }
        raise RuntimeError(
            f"V29 hierarchical representation is not finite: {missing_counts}"
        )
    return result


def shrink_metric(
    l2_values: pd.Series,
    l3_values: pd.Series,
    *,
    l3_count: pd.Series,
) -> pd.Series:
    l2 = pd.to_numeric(l2_values, errors="coerce")
    l3 = pd.to_numeric(l3_values, errors="coerce")
    count = pd.to_numeric(l3_count, errors="coerce").fillna(0.0).clip(
        lower=0.0
    )
    weight = count / (count + float(L3_SHRINKAGE_PSEUDO_PEERS))
    both = l2.notna() & l3.notna()
    only_l3 = l2.isna() & l3.notna()
    value = pd.Series(0.0, index=l2.index, dtype=float)
    value.loc[both] = (
        weight.loc[both] * l3.loc[both]
        + (1.0 - weight.loc[both]) * l2.loc[both]
    )
    value.loc[only_l3] = l3.loc[only_l3]
    value.loc[l2.notna() & l3.isna()] = l2.loc[
        l2.notna() & l3.isna()
    ]
    return value


def audit_hierarchical_feature_coverage(
    features: pd.DataFrame,
    candidate_index: pd.DataFrame,
) -> dict[str, Any]:
    required = {*IDENTITY_COLUMNS, "fold", *V29_FEATURE_COLUMNS}
    missing = sorted(required - set(features.columns))
    duplicate_identity = bool(
        features.duplicated(list(IDENTITY_COLUMNS)).any()
    )
    expected = candidate_index.loc[
        :, [*IDENTITY_COLUMNS, "fold"]
    ].drop_duplicates()
    actual = features.loc[
        :, [*IDENTITY_COLUMNS, "fold"]
    ].drop_duplicates()
    identity_match = bool(
        len(actual) == len(expected)
        and actual.merge(
            expected.assign(_expected=1),
            on=[*IDENTITY_COLUMNS, "fold"],
            how="outer",
            indicator=True,
        )["_merge"].eq("both").all()
    )
    if missing:
        finite = pd.Series(False, index=features.index, dtype=bool)
    else:
        numeric = features.loc[:, V29_FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite = numeric.notna().all(axis=1) & np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all(axis=1)
    coverage = float(finite.mean()) if len(features) else 0.0
    no_peer_rows = (
        int(
            pd.to_numeric(
                features.get("v29_peer_no_peer_context"),
                errors="coerce",
            )
            .eq(1.0)
            .sum()
        )
        if "v29_peer_no_peer_context" in features
        else 0
    )
    return {
        "expected_candidate_rows": int(len(expected)),
        "feature_rows": int(len(features)),
        "missing_columns": missing,
        "duplicate_identity": duplicate_identity,
        "candidate_identity_match": identity_match,
        "finite_feature_rows": int(finite.sum()),
        "finite_feature_coverage": coverage,
        "minimum_finite_feature_coverage": 1.0,
        "no_peer_context_rows": no_peer_rows,
        "coverage_passed": bool(
            not missing
            and not duplicate_identity
            and identity_match
            and coverage == 1.0
        ),
    }

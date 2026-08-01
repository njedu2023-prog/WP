from __future__ import annotations

import copy
from pathlib import Path

import pytest

from wp.v3.contracts import load_v3_config
from wp.v3.frozen_candidate import (
    FROZEN_CANDIDATE_ID,
    FROZEN_SHADOW_STATUS,
    FrozenCandidateError,
    load_frozen_candidate,
    verify_frozen_candidate,
    verify_runtime_contract,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "wp_v15_frozen_shadow_candidate.json"
CONFIG_PATH = ROOT / "config" / "wp_v3.yml"


def test_frozen_manifest_verifies_exact_evidence_and_policy() -> None:
    manifest = load_frozen_candidate(MANIFEST_PATH)

    assert manifest["candidate_id"] == FROZEN_CANDIDATE_ID
    assert manifest["production_authorized"] is False
    assert manifest["shadow_contract"]["status"] == FROZEN_SHADOW_STATUS


def test_any_manifest_mutation_breaks_integrity() -> None:
    manifest = load_frozen_candidate(MANIFEST_PATH)
    mutated = copy.deepcopy(manifest)
    mutated["candidate_spec"]["entry_policy"]["probability_min"] = 0.53

    with pytest.raises(FrozenCandidateError, match="verification failed"):
        verify_frozen_candidate(mutated)


def test_v15_cannot_inherit_v40_runtime_or_shadow_clock() -> None:
    manifest = load_frozen_candidate(MANIFEST_PATH)
    config = load_v3_config(CONFIG_PATH)

    with pytest.raises(
        FrozenCandidateError,
        match="entry_execution_deadline",
    ):
        verify_runtime_contract(manifest, config)


def test_forward_evidence_is_not_mislabeled_as_shadow_time() -> None:
    manifest = load_frozen_candidate(MANIFEST_PATH)
    shadow = manifest["shadow_contract"]

    assert shadow["started_trade_date"] is None
    assert shadow["verified_candidates"] == 0
    assert shadow["clock_inheritance_allowed"] is False
    assert shadow["production_promotion_allowed"] is False

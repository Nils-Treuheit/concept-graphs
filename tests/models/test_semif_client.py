"""Core contract checks for the optional SemIf relation adapter."""

from __future__ import annotations

import pytest

from conceptgraph.utils.semif_client import build_relation_row, proposal_from_score


def test_relation_row_preserves_declared_candidate_order_and_unknown() -> None:
    row = build_relation_row(
        "mug at x=320; plate at x=476",
        {"left_of": "mug is left", "unknown": "insufficient evidence"},
    )
    assert [option["id"] for option in row["options"]] == ["left_of", "unknown"]


def test_relation_row_requires_unknown_abstention_candidate() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_relation_row("observed pair", {"left_of": "left", "right_of": "right"})


def test_proposal_maps_scores_to_relation_ids_and_marks_advisory() -> None:
    result = proposal_from_score(
        {
            "option_ids": ["left_of", "unknown"],
            "probabilities": [0.8, 0.2],
            "forward_seconds": 1.2,
        },
        {"source": "Qwen/Qwen3.5-4B"},
    )
    assert result["choice"] == "left_of"
    assert result["probabilities"] == {"left_of": 0.8, "unknown": 0.2}
    assert result["conditional_probability"] == 0.8
    assert result["advisory_only"] is True


def test_proposal_rejects_misaligned_score_arrays() -> None:
    with pytest.raises(ValueError, match="invalid option IDs"):
        proposal_from_score(
            {"option_ids": ["left_of", "unknown"], "probabilities": [0.8]}, {}
        )

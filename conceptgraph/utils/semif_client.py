"""SemIf direct-choice adapter for advisory ConceptGraphs relation proposals."""

from __future__ import annotations

import math
from typing import Any


class SemIfError(RuntimeError):
    """Raised when a SemIf relation proposal cannot be produced."""


def build_relation_row(
    state: str,
    candidates: dict[str, str],
    *,
    question: str = "Which spatial relation is supported by the observations?",
    request_id: str = "conceptgraphs-relation",
) -> dict[str, Any]:
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be non-empty text")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(candidates, dict) or not 2 <= len(candidates) <= 8:
        raise ValueError("provide 2 to 8 named relation candidates")
    if "unknown" not in candidates:
        raise ValueError("candidates must include unknown")
    if any(not isinstance(key, str) or not key.strip() for key in candidates):
        raise ValueError("candidate names must be non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in candidates.values()):
        raise ValueError("candidate descriptions must be non-empty strings")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty text")
    return {
        "id": request_id,
        "state": state,
        "question": question,
        "options": [
            {"id": key, "description": description}
            for key, description in candidates.items()
        ],
    }


def proposal_from_score(result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    option_ids = result.get("option_ids")
    scores = result.get("probabilities")
    if (
        not isinstance(option_ids, list)
        or len(option_ids) < 2
        or len(option_ids) != len(scores or [])
        or len(set(option_ids)) != len(option_ids)
        or any(not isinstance(item, str) for item in option_ids)
    ):
        raise ValueError("SemIf returned invalid option IDs or probabilities")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores):
        raise ValueError("SemIf returned a non-finite probability")
    probabilities = dict(zip(option_ids, scores))
    choice = max(probabilities, key=probabilities.__getitem__)
    return {
        "choice": choice,
        "conditional_probability": probabilities[choice],
        "probabilities": probabilities,
        "model": metadata,
        "readout": result.get("readout"),
        "probability_status": result.get("probability_status"),
        "timing": {
            key: result[key]
            for key in ("forward_seconds", "total_seconds", "input_tokens")
            if key in result
        },
        "prompt_sha256": result.get("prompt_sha256"),
        "advisory_only": True,
    }


class SemIfRelationClient:
    """Load one SemIf model and reuse it for relation-choice requests."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3.5-4B",
        revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        try:
            from semif_phase1.core import load_causal_model
        except ImportError as exc:
            raise SemIfError(
                "SemIf is not installed; add its src directory to PYTHONPATH"
            ) from exc
        self.model, self.tokenizer, self.metadata = load_causal_model(
            model, revision, device=device, dtype=dtype
        )
        self.model.eval()

    def choose_relation(
        self,
        state: str,
        candidates: dict[str, str],
        *,
        question: str = "Which spatial relation is supported by the observations?",
        request_id: str = "conceptgraphs-relation",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        row = build_relation_row(
            state, candidates, question=question, request_id=request_id
        )
        from semif_phase1.direct import score

        try:
            result = score(self.model, self.tokenizer, row, self.metadata, max_tokens)
            return proposal_from_score(result, self.metadata)
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise SemIfError(f"SemIf relation scoring failed: {exc}") from exc
            raise SemIfError(f"SemIf relation scoring failed: {exc}") from exc

"""Small Open-Jev adapter for advisory scene-graph relation choices."""

from __future__ import annotations

import requests


class OpenJevError(RuntimeError):
    """Raised when Open-Jev is unavailable or returns an invalid answer."""


def choose_relation(
    state: str,
    candidates: dict[str, str],
    *,
    endpoint: str = "http://127.0.0.1:8792",
    timeout: float = 30,
) -> dict:
    """Return a typed, non-authoritative relation proposal.

    Always include an `unknown` option so the model can abstain. No graph is
    changed by this function.
    """
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be non-empty text")
    if not isinstance(candidates, dict) or not 2 <= len(candidates) <= 8:
        raise ValueError("provide 2 to 8 named relation candidates")
    if "unknown" not in candidates:
        raise ValueError("candidates must include unknown")
    if any(not isinstance(k, str) or not k.strip() for k in candidates):
        raise ValueError("candidate names must be non-empty strings")
    if any(not isinstance(v, str) for v in candidates.values()):
        raise ValueError("candidate descriptions must be strings")

    payload = {
        "model": "open-jev",
        "state": state,
        "questions": {
            "relation": {
                "type": "choice",
                "instructions": (
                    "Choose the spatial relation supported by the supplied "
                    "observations. Choose unknown when evidence is insufficient."
                ),
                "criteria": candidates,
            }
        },
    }
    try:
        response = requests.post(
            endpoint.rstrip("/") + "/v1/systemone",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        answer = body["answers"]["relation"]
        choice = answer["choice"]
        if answer.get("type") != "choice" or choice not in candidates:
            raise ValueError("response choice is outside the declared options")
        return {
            "choice": choice,
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities", {}),
            "model": body.get("model"),
            "usage": body.get("usage"),
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise OpenJevError(f"Open-Jev relation request failed: {exc}") from exc

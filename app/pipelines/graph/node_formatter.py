from __future__ import annotations

from typing import Any, Dict

from app.pipelines.graph.state import AgentState
from app.pipelines.graph.node_type1 import deterministic_type1_solver
from app.schemas import PredictRequest
from app.utils.normalize import coerce_response


def formatter_node(state: AgentState) -> Dict[str, Any]:
    """Node that normalizes and formats the final answer/unit using coerce_response."""
    req = PredictRequest(
        query_id=state["query_id"],
        type=state["qtype"],
        query=state["query"],
        premises=state["premises"],
        options=state["options"],
    )

    answer = state.get("llm_answer", "")
    if req.type == "type1":
        verified = state.get("verified_answer", "").strip()
        if verified:
            answer = verified

    raw = {
        "answer": answer,
        "unit": state.get("unit", ""),
        "explanation": state.get("explanation", ""),
        "premises_used": state.get("premises_used", []),
        "reasoning": state.get("reasoning", {}),
    }

    if req.type == "type1" and state.get("deterministic_locked", False):
        return {
            "final_answer": str(raw["answer"]).strip(),
            "final_unit": "",
            "final_explanation": str(raw["explanation"]).strip() or "Deterministic rule solver result.",
            "final_premises_used": list(raw.get("premises_used", [])),
            "final_reasoning": raw.get("reasoning", {}),
        }

    if req.type == "type1" and state.get("needs_retry", False) and not state.get("verified_answer", "").strip():
        fallback = deterministic_type1_solver(req)
        if fallback:
            return {
                "final_answer": str(fallback["answer"]).strip(),
                "final_unit": "",
                "final_explanation": str(fallback["explanation"]).strip(),
                "final_premises_used": list(fallback.get("premises_used", [])),
                "final_reasoning": fallback.get("reasoning", {}),
            }

    # Coerce using our normalization helpers
    coerced = coerce_response(req, raw)

    return {
        "final_answer": coerced.answer,
        "final_unit": coerced.unit,
        "final_explanation": coerced.explanation,
        "final_premises_used": coerced.premises_used,
        "final_reasoning": coerced.reasoning.model_dump() if hasattr(coerced.reasoning, "model_dump") else coerced.reasoning,
    }

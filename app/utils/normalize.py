from __future__ import annotations

import re
from typing import Any

from app.schemas import PredictRequest, PredictResponseItem


_ASCII_UNIT_REPLACEMENTS = {
    "Ω": "ohm",
    "Ω": "ohm",
    "\u03a9": "ohm",
    "μ": "u",
    "µ": "u",
    "°": "degree",
}


def normalize_unit(unit: Any) -> str:
    value = "" if unit is None else str(unit).strip()
    for src, dst in _ASCII_UNIT_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.replace(" ", "")
    return value


def _normalize_choice(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def coerce_response(req: PredictRequest, raw: dict[str, Any]) -> PredictResponseItem:
    answer = str(raw.get("answer", "")).strip()
    explanation = str(raw.get("explanation", "")).strip()
    if not explanation:
        explanation = "The system produced the final answer using the available query fields."

    reasoning = raw.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        reasoning = {"type": "raw", "steps": [str(reasoning)]}

    if req.type == "type1":
        # The competition requires an exact option string when options is non-empty.
        if req.options:
            normalized = {_normalize_choice(opt): opt for opt in req.options}
            if _normalize_choice(answer) in normalized:
                answer = normalized[_normalize_choice(answer)]
            elif "Uncertain" in req.options:
                answer = "Uncertain"
            else:
                answer = req.options[0]

        premises_used_raw = raw.get("premises_used", [])
        premises_used: list[int] = []
        if isinstance(premises_used_raw, list):
            for item in premises_used_raw:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(req.premises) and idx not in premises_used:
                    premises_used.append(idx)

        return PredictResponseItem(
            query_id=req.query_id,
            answer=answer or ("Uncertain" if not req.options else req.options[0]),
            unit="",
            explanation=explanation,
            premises_used=premises_used,
            reasoning=reasoning,
        )

    # Type 2: answer should be numerical/string value only; unit goes separately.
    return PredictResponseItem(
        query_id=req.query_id,
        answer=answer or "0",
        unit=normalize_unit(raw.get("unit", "")),
        explanation=explanation,
        premises_used=[],
        reasoning=reasoning,
    )

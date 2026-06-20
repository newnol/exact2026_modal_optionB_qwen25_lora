from __future__ import annotations

import re
from typing import Any

from app.logging_utils import log_entry
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


def _reasoning_text(raw: dict[str, Any]) -> str:
    """Extract only reasoning steps (FOL trace) for precise premise matching."""
    reasoning = raw.get("reasoning")
    if isinstance(reasoning, dict):
        steps = reasoning.get("steps")
        if isinstance(steps, list):
            return " ".join(_norm(str(s)) for s in steps)
    return ""


def _all_text(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    exp = raw.get("explanation")
    if exp:
        parts.append(_norm(str(exp)))
    reasoning = raw.get("reasoning")
    if isinstance(reasoning, dict):
        steps = reasoning.get("steps")
        if isinstance(steps, list):
            parts.extend(_norm(str(s)) for s in steps)
    return " ".join(parts)


def _norm(text: str) -> str:
    """Normalize text for matching: lowercase, underscores→spaces."""
    return text.lower().replace("_", " ").replace("-", " ")


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if keyword (handling underscore/space equivalence) appears in text."""
    nk = _norm(keyword)
    nt = _norm(text)
    return nk in nt


def audit_premises_used(
    req: PredictRequest,
    raw: dict[str, Any],
    premises_used: list[int],
) -> list[int]:
    answer = str(raw.get("answer", "")).strip()
    text = _all_text(raw)
    premises = req.premises
    result = set(premises_used)

    # --- Rule 1: Uncertain + denial premise → only denial premise ---
    if _normalize_choice(answer) == "uncertain":
        denial_idx = [
            i for i, p in enumerate(premises)
            if any(kw in p.lower() for kw in [
                "no premise", "nothing", "not stated",
                "does not state", "not mention", "no information",
            ])
        ]
        if denial_idx:
            return denial_idx

    # --- Rule 2: Include premise referenced by index in text ---
    for i in range(len(premises)):
        if f"premise {i}" in text or f"(premise {i})" in text:
            result.add(i)

    # --- Rule 3: If answer is a number, prefer premise containing that number ---
    num_match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if num_match:
        ans_num = num_match.group()
        for i, p in enumerate(premises):
            if ans_num in p:
                result.add(i)

    # --- Use reasoning steps (FOL trace) as the primary matching target ---
    rtext = _reasoning_text(raw) or _all_text(raw)

    # Words too common across premises to be useful for matching
    _COMMON = frozenset({
        "that", "this", "with", "from", "have", "been",
        "then", "than", "were", "will", "would", "could",
        "should", "must", "shall", "into", "over", "such",
        "each", "also", "after", "before", "about", "between",
        "their", "there", "when", "what", "does", "doing",
        "some", "more", "most", "other", "only", "very",
        "just", "they", "them", "who", "which", "whose",
        "may", "can", "has", "had", "not", "are", "all",
        "every", "where", "while", "these", "those",
        "researcher", "study", "alpha", "asha",
    })

    # --- Rule 4: Cross-check each premise against reasoning steps ---
    referenced: set[int] = set()
    premise_conclusions: list[str | None] = []

    for i, p in enumerate(premises):
        np = _norm(p)
        low = p.lower()
        premise_conclusions.append(None)

        # For If-Then rules: check if distinctive conclusion words appear in reasoning
        if low.startswith("if ") and "then " in low:
            conclusion = low.split("then", 1)[-1].strip().rstrip(".").rstrip(",")
            premise_conclusions[i] = conclusion
            conc_distinct = {
                w for w in re.findall(r"[a-zA-Z][a-zA-Z]{2,}", _norm(conclusion))
                if w not in _COMMON
            }
            if not conc_distinct:
                continue
            r_matches = sum(1 for w in conc_distinct if re.search(rf"\b{re.escape(w)}\b", rtext))
            if r_matches >= 1 and r_matches / len(conc_distinct) >= 0.3:
                referenced.add(i)

        # For factual premises: check distinctive words against reasoning steps
        else:
            distinctive = {
                w for w in re.findall(r"[a-zA-Z][a-zA-Z]{3,}", np)
                if w not in _COMMON
            }
            if not distinctive:
                continue
            r_matches = sum(1 for kw in distinctive if re.search(rf"\b{re.escape(kw)}\b", rtext))
            if r_matches >= 1 and r_matches / len(distinctive) >= 0.35:
                referenced.add(i)

    # --- Rule 5: Chain rules via distinctive word overlap ---
    for i, conc in enumerate(premise_conclusions):
        if conc is None or i in referenced:
            continue
        conc_distinct = {
            w for w in re.findall(r"[a-zA-Z][a-zA-Z]{2,}", _norm(conc))
            if w not in _COMMON
        }
        if not conc_distinct:
            continue
        r_matches = sum(1 for w in conc_distinct if re.search(rf"\b{re.escape(w)}\b", rtext))
        if r_matches >= 1:
            referenced.add(i)

    # --- Rule 6: Filter result ---
    final: set[int] = set()
    for i in result:
        if i in referenced:
            final.add(i)
        elif premises[i].lower().startswith("if "):
            final.add(i)  # keep rules implicitly
        elif f"premise {i}" in text or f"(premise {i})" in text:
            final.add(i)

    # --- Rule 7: Add high-confidence premises the model missed ---
    for i in referenced:
        if i not in final:
            final.add(i)

    if not final:
        final = result  # fallback

    final_sorted = sorted(final)
    if final_sorted != sorted(premises_used):
        log_entry({
            "event": "premise_audit",
            "query_id": req.query_id,
            "model_gave": sorted(premises_used),
            "auditor_fixed": final_sorted,
        })

    return final_sorted


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

        premises_used = audit_premises_used(req, raw, premises_used)

        return PredictResponseItem(
            query_id=req.query_id,
            answer=answer or ("Uncertain" if not req.options else req.options[0]),
            unit="",
            explanation=explanation,
            premises_used=premises_used,
            reasoning=reasoning,
        )

    # Type 2: answer should be numerical/string value only; unit goes separately.
    processed_ans, processed_unit = _process_type2_answer(answer or "0", raw.get("unit", ""), req.query)
    return PredictResponseItem(
        query_id=req.query_id,
        answer=processed_ans,
        unit=normalize_unit(processed_unit),
        explanation=explanation,
        premises_used=[],
        reasoning=reasoning,
    )


def _process_single_type2_value(val_str: str, unit: str, query: str) -> tuple[str, str]:
    # Try parsing as float
    numeric_val = None
    try:
        # Remove spaces, commas, and standard unicode characters
        clean_val = re.sub(r"\s+", "", val_str)
        numeric_val = float(clean_val)
    except ValueError:
        return val_str, unit

    # 3. Unit Conversion & Scaling from SI units to target units
    q_lower = query.lower()
    has_micro = any(x in q_lower for x in ["micro", "μ", "\u03bc", "u"])
    has_milli = any(x in q_lower for x in ["milli", "m"])
    has_nano = any(x in q_lower for x in ["nano", "n"])
    has_pico = any(x in q_lower for x in ["pico", "p"])

    # Determine implied quantity from query
    inferred_qty = None
    if "capacit" in q_lower or "capacitor" in q_lower:
        inferred_qty = "capacitance"
    elif "charge" in q_lower:
        inferred_qty = "charge"
    elif "current" in q_lower:
        inferred_qty = "current"
    elif "energy" in q_lower or "work" in q_lower or "heat" in q_lower:
        inferred_qty = "energy"

    target_unit = unit

    if target_unit == "F" or (not target_unit and inferred_qty == "capacitance"):
        # standard capacitance is in uF, nF, or pF
        if numeric_val < 1e-9:
            numeric_val *= 1e12
            target_unit = "pF"
        elif numeric_val < 1e-6:
            numeric_val *= 1e9
            target_unit = "nF"
        elif numeric_val < 1.0:
            numeric_val *= 1e6
            target_unit = "uF"
    elif target_unit == "C" or (not target_unit and inferred_qty == "charge"):
        if numeric_val < 1e-9:
            numeric_val *= 1e12
            target_unit = "pC"
        elif numeric_val < 1e-6:
            numeric_val *= 1e9
            target_unit = "nC"
        elif numeric_val < 1e-3:
            numeric_val *= 1e6
            target_unit = "uC"
        elif numeric_val < 1.0:
            numeric_val *= 1e3
            target_unit = "mC"
    elif target_unit == "J" or (not target_unit and inferred_qty == "energy"):
        if "uj" in q_lower or "microjoule" in q_lower or (has_micro and numeric_val < 1e-3):
            numeric_val *= 1e6
            target_unit = "uJ"
        elif "mj" in q_lower or "millijoule" in q_lower or (has_milli and numeric_val < 1.0):
            numeric_val *= 1e3
            target_unit = "mJ"
    elif target_unit == "A" or (not target_unit and inferred_qty == "current"):
        if "ma" in q_lower or "milliampere" in q_lower or (has_milli and numeric_val < 1.0):
            numeric_val *= 1e3
            target_unit = "mA"
        elif "ua" in q_lower or "microampere" in q_lower or (has_micro and numeric_val < 1e-3):
            numeric_val *= 1e6
            target_unit = "uA"

    # 4. Rounding and Formatting
    # Round to 6 decimal places to remove precision noise
    rounded = round(numeric_val, 6)
    
    # Format whole numbers as integers
    if rounded.is_integer():
        formatted_ans = str(int(rounded))
    else:
        abs_val = abs(rounded)
        if abs_val < 0.01 or abs_val >= 5000:
            # format as scientific notation with 2 decimal places (3 sig figs)
            s_notation = f"{rounded:.2e}"
            match = re.match(r"(-?\d+\.\d+)e([+-]\d+)", s_notation)
            if match:
                coeff = match.group(1)
                exp = int(match.group(2))
                formatted_ans = f"{coeff} × 10^{exp}"
            else:
                formatted_ans = str(rounded)
        else:
            formatted_ans = f"{rounded:.2f}".rstrip("0").rstrip(".")

    return formatted_ans, target_unit


def _process_type2_answer(answer: str, unit: str, query: str) -> tuple[str, str]:
    import json
    # 1. Parse JSON fallback if the answer is a raw JSON string
    answer = answer.strip()
    if answer.startswith("{") and answer.endswith("}"):
        try:
            parsed = json.loads(answer)
            val = parsed.get("value") or parsed.get("answer")
            if val is not None:
                answer = str(val).strip()
            if not unit:
                unit = normalize_unit(parsed.get("unit", ""))
        except Exception:
            pass

    # 2. Extract multiple values if separated by semicolon (e.g. "0; 0")
    if ";" in answer:
        parts = [p.strip() for p in answer.split(";")]
        formatted_parts = []
        for part in parts:
            f_val, _ = _process_single_type2_value(part, unit, query)
            formatted_parts.append(f_val)
        return "; ".join(formatted_parts), unit

    # Otherwise, it's a single value
    return _process_single_type2_value(answer, unit, query)

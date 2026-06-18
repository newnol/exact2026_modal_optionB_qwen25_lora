from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from app.pipelines.graph.state import AgentState
from app.schemas import PredictRequest
from app.utils.json_utils import extract_json_object


def _first_float_after(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def deterministic_physics_solver(query: str) -> dict[str, Any] | None:
    """Small conservative fast path for common/public patterns."""
    q = query.replace("Ω", "ohm").replace("Ω", "ohm").replace("×", "x")

    if re.search(r"parallel", q, re.IGNORECASE) and re.search(r"total\s+current|current", q, re.IGNORECASE):
        nums = re.findall(r"R\s*\d*\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ohm|Ω|Ω)", q, flags=re.IGNORECASE)
        voltage = _first_float_after(r"([0-9]+(?:\.[0-9]+)?)\s*V", q)
        if len(nums) >= 2 and voltage is not None:
            r1, r2 = float(nums[0]), float(nums[1])
            req = 1.0 / (1.0 / r1 + 1.0 / r2)
            current = voltage / req
            return {
                "answer": f"{current:g}",
                "unit": "A",
                "explanation": f"For two resistors in parallel, Req = {req:g} ohm, so I = V/Req = {voltage:g}/{req:g} = {current:g} A.",
                "premises_used": [],
                "reasoning": {
                    "type": "compute",
                    "steps": [
                        f"1/Req = 1/{r1:g} + 1/{r2:g}",
                        f"Req = {req:g} ohm",
                        f"I = {voltage:g}/{req:g} = {current:g} A",
                    ],
                },
            }

    return None


SYSTEM_PROMPT_TYPE2_DIRECT = """You solve physics problems for EXACT 2026.
Return ONLY one valid JSON object with keys: answer, unit, explanation, premises_used, reasoning.
Rules:
- answer MUST contain the numerical value or short text only, without the unit.
- unit MUST be ASCII only and strictly follow the Unit Dictionary below.
- premises_used MUST be [] for type2.
- explanation MUST be non-empty.
- reasoning should be an object like {"type":"cot","steps":[...]}.
- Do not include markdown or extra text outside JSON.

Unit Dictionary Guidelines:
- Capacitance: "F", "uF", "nF", "pF"
- Electric Charge: "C", "uC", "nC", "pC"
- Resistance / Impedance: "ohm" (use "ohm", NOT the omega symbol "Ω" or "Ω")
- Current: "A", "mA", "uA"
- Potential Difference / Voltage: "V"
- Electric Field: "V/m" or "N/C"
- Work / Energy: "J", "mJ"
- Power: "W"
- Power factor (dimensionless): "" (empty string)
- Relative ratio, change factor, or percentage: "" (empty string, e.g. "decreases by half" or "2" or "0.5")
- Direction, qualitative answers: "" (empty string)
- If a value has no unit (dimensionless), "unit" MUST be an empty string "".
"""

SYSTEM_PROMPT_TYPE2_CODE = """You are the Type-2 physics code generator for EXACT 2026.
Return ONLY one valid JSON object with keys: python_code, unit, explanation.
Rules:
- python_code must be standalone Python that computes the final numeric answer and prints ONLY the final value.
- You may use math and sympy as sp.
- Do not read files, use network, spawn processes, or print explanations.
- unit MUST be ASCII only and strictly follow the Unit Dictionary below.
- explanation is a short natural-language plan, not the full final explanation.
- Do not include markdown or extra text outside JSON.

Unit Dictionary Guidelines:
- Capacitance: "F", "uF", "nF", "pF"
- Electric Charge: "C", "uC", "nC", "pC"
- Resistance / Impedance: "ohm" (use "ohm", NOT the omega symbol "Ω" or "Ω")
- Current: "A", "mA", "uA"
- Potential Difference / Voltage: "V"
- Electric Field: "V/m" or "N/C"
- Work / Energy: "J", "mJ"
- Power: "W"
- Power factor (dimensionless): "" (empty string)
- Relative ratio, change factor, or percentage: "" (empty string, e.g. "decreases by half" or "2" or "0.5")
- Direction, qualitative answers: "" (empty string)
- If a value has no unit (dimensionless), "unit" MUST be an empty string "".
"""

def build_type2_direct_prompt(req: PredictRequest) -> str:
    return json.dumps(
        {
            "query_id": req.query_id,
            "type": req.type,
            "query": req.query,
            "required_output_shape": {
                "answer": "number or short text only, no unit",
                "unit": "ASCII unit, for example A, V, ohm, V/m, J, W, uF, nC",
                "explanation": "non-empty explanation",
                "premises_used": [],
                "reasoning": {"type": "cot", "steps": ["step 1", "step 2"]},
            },
        },
        ensure_ascii=False,
    )


def build_type2_code_prompt(req: PredictRequest) -> str:
    return json.dumps(
        {
            "query_id": req.query_id,
            "query": req.query,
            "task": "Generate standalone Python/SymPy code to compute the final answer. Print only the final numeric value.",
            "required_output_shape": {
                "python_code": "standalone Python code that prints only the final value",
                "unit": "ASCII answer unit",
                "explanation": "brief computation plan",
            },
        },
        ensure_ascii=False,
    )


async def llm_type2_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node that generates SymPy physics solver code."""
    llm_client = config.get("configurable", {}).get("type2_llm")
    if not llm_client:
        raise ValueError("type2_llm client must be provided in config['configurable']")

    req = PredictRequest(
        query_id=state["query_id"],
        type=state["qtype"],
        query=state["query"],
        premises=state["premises"],
        options=state["options"],
    )

    try:
        text = await llm_client.chat_json(
            SYSTEM_PROMPT_TYPE2_CODE,
            build_type2_code_prompt(req),
            query_id=state["query_id"],
            pipeline="type2_graph_code",
        )
        raw = extract_json_object(text)
    except Exception as exc:
        return {
            "explanation": f"LLM Type 2 code call failed: {exc}",
            "generated_code": "",
        }

    return {
        "generated_code": str(raw.get("python_code", "")).strip(),
        "unit": str(raw.get("unit", "")).strip(),
        "explanation": str(raw.get("explanation", "")).strip(),
    }


async def llm_type2_direct_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Fallback node that predicts physics answer directly without code execution."""
    llm_client = config.get("configurable", {}).get("type2_llm")
    if not llm_client:
        raise ValueError("type2_llm client must be provided in config['configurable']")

    req = PredictRequest(
        query_id=state["query_id"],
        type=state["qtype"],
        query=state["query"],
        premises=state["premises"],
        options=state["options"],
    )

    try:
        text = await llm_client.chat_json(
            SYSTEM_PROMPT_TYPE2_DIRECT,
            build_type2_direct_prompt(req),
            query_id=state["query_id"],
            pipeline="type2_graph_direct",
        )
        raw = extract_json_object(text)
    except Exception as exc:
        return {
            "llm_answer": "0",
            "explanation": f"LLM Type 2 direct fallback failed: {exc}",
        }

    return {
        "llm_answer": str(raw.get("answer", "")).strip(),
        "unit": str(raw.get("unit", "")).strip(),
        "explanation": str(raw.get("explanation", "")).strip(),
        "premises_used": [],
        "reasoning": raw.get("reasoning", {}),
    }

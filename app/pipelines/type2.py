from __future__ import annotations

import json
import re
from typing import Any

from app.llm_client import VLLMClient
from app.schemas import PredictRequest, PredictResponseItem, fallback_response
from app.utils.json_utils import extract_json_object
from app.utils.normalize import coerce_response
from app.utils.sandbox import last_stdout_value, run_python_code


SYSTEM_PROMPT_TYPE2_DIRECT = """You solve physics problems for EXACT 2026.
Return ONLY one valid JSON object with keys: answer, unit, explanation, premises_used, reasoning.
Rules:
- answer MUST contain the numerical value or short text only, without the unit.
- unit MUST be ASCII only, e.g. A, V, ohm, V/m, J, W, uF, nC.
- premises_used MUST be [] for type2.
- explanation MUST be non-empty.
- reasoning should be an object like {"type":"cot","steps":[...]}.
- Do not include markdown or extra text outside JSON.
"""

SYSTEM_PROMPT_TYPE2_CODE = """You are the Type-2 physics code generator for EXACT 2026.
Return ONLY one valid JSON object with keys: python_code, unit, explanation.
Rules:
- python_code must be standalone Python that computes the final numeric answer and prints ONLY the final value.
- You may use math and sympy as sp.
- Do not read files, use network, spawn processes, or print explanations.
- unit must be ASCII only, e.g. A, V, ohm, V/m, J, W, uF, nC.
- explanation is a short natural-language plan, not the full final explanation.
- Do not include markdown or extra text outside JSON.
"""

SYSTEM_PROMPT_TYPE2_EXPLAIN = """You explain a verified physics computation for EXACT 2026.
Return ONLY one valid JSON object with keys: explanation, reasoning.
Rules:
- explanation must be non-empty and concise.
- reasoning must be an object like {"type":"sandbox_verified","steps":[...]}.
- Do not change the provided answer or unit.
- Do not include markdown or extra text outside JSON.
"""


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


def build_type2_explain_prompt(req: PredictRequest, answer: str, unit: str, plan: str, code: str) -> str:
    return json.dumps(
        {
            "query_id": req.query_id,
            "query": req.query,
            "verified_answer": answer,
            "verified_unit": unit,
            "computation_plan": plan,
            "python_code": code,
            "required_output_shape": {
                "explanation": "concise explanation using the verified result",
                "reasoning": {"type": "sandbox_verified", "steps": ["step 1", "step 2"]},
            },
        },
        ensure_ascii=False,
    )


async def _sandbox_pipeline(req: PredictRequest, llm: VLLMClient) -> dict[str, Any] | None:
    text = await llm.chat_json(SYSTEM_PROMPT_TYPE2_CODE, build_type2_code_prompt(req))
    code_obj = extract_json_object(text)
    python_code = str(code_obj.get("python_code", "")).strip()
    unit = str(code_obj.get("unit", "")).strip()
    plan = str(code_obj.get("explanation", "")).strip()

    result = run_python_code(python_code, timeout_seconds=3.0)
    if not result.ok:
        return None

    answer = last_stdout_value(result.stdout)
    if not answer:
        return None

    explanation = plan or "The answer was computed by executing the generated physics solver."
    reasoning: dict[str, Any] = {
        "type": "sandbox_verified",
        "steps": [
            "Generated a standalone Python/SymPy solver.",
            "Executed the solver in a short-timeout subprocess sandbox.",
            f"The sandbox printed the verified result: {answer} {unit}".strip(),
        ],
    }

    try:
        exp_text = await llm.chat_json(
            SYSTEM_PROMPT_TYPE2_EXPLAIN,
            build_type2_explain_prompt(req, answer=answer, unit=unit, plan=plan, code=python_code),
        )
        exp_obj = extract_json_object(exp_text)
        explanation = str(exp_obj.get("explanation") or explanation).strip() or explanation
        if isinstance(exp_obj.get("reasoning"), dict):
            reasoning = exp_obj["reasoning"]
    except Exception:
        pass

    return {
        "answer": answer,
        "unit": unit,
        "explanation": explanation,
        "premises_used": [],
        "reasoning": reasoning,
    }


async def solve_type2(
    req: PredictRequest,
    llm: VLLMClient,
    mock_mode: bool = False,
    use_llm_fallback: bool = True,
) -> PredictResponseItem:
    fast = deterministic_physics_solver(req.query)
    if fast is not None:
        return coerce_response(req, fast)

    if mock_mode:
        raw = {
            "answer": "0",
            "unit": "",
            "explanation": "MOCK_MODE response. Connect vLLM to generate real physics answers.",
            "premises_used": [],
            "reasoning": {"type": "mock", "steps": ["Mock response generated for API-format testing."]},
        }
        return coerce_response(req, raw)

    if not use_llm_fallback:
        return fallback_response(req, "No deterministic Type 2 solver matched and TYPE2_FALLBACK_TO_LLM=false.")

    try:
        sandbox_raw = await _sandbox_pipeline(req, llm)
        if sandbox_raw is not None:
            return coerce_response(req, sandbox_raw)
    except Exception:
        # Fall through to direct answer generation.
        pass

    try:
        text = await llm.chat_json(SYSTEM_PROMPT_TYPE2_DIRECT, build_type2_direct_prompt(req))
        raw = extract_json_object(text)
        return coerce_response(req, raw)
    except Exception as exc:
        return fallback_response(req, f"Type 2 pipeline failed safely: {exc}")

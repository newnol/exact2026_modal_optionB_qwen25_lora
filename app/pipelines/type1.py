from __future__ import annotations

import json

from app.llm_client import VLLMClient
from app.schemas import PredictRequest, PredictResponseItem, fallback_response
from app.utils.json_utils import extract_json_object
from app.utils.normalize import coerce_response


SYSTEM_PROMPT_TYPE1 = """You solve logic-based educational questions for EXACT 2026.
Return ONLY one valid JSON object with keys: answer, unit, explanation, premises_used, reasoning.
Rules:
- If options is non-empty, answer MUST exactly equal one of the provided options.
- unit MUST be an empty string for type1.
- premises_used MUST contain only 0-based indices of premises actually used.
- explanation MUST be non-empty.
- reasoning should be an object like {"type":"fol","steps":[...]}.
- Do not include markdown or extra text outside JSON.
"""


def build_type1_prompt(req: PredictRequest) -> str:
    return json.dumps(
        {
            "query_id": req.query_id,
            "type": req.type,
            "query": req.query,
            "premises": req.premises,
            "options": req.options,
            "required_output_shape": {
                "answer": "exact option if options non-empty, otherwise short answer",
                "unit": "",
                "explanation": "non-empty explanation",
                "premises_used": "list of 0-based premise indices used",
                "reasoning": {"type": "fol", "steps": ["step 1", "step 2"]},
            },
        },
        ensure_ascii=False,
    )


async def solve_type1(req: PredictRequest, llm: VLLMClient, mock_mode: bool = False) -> PredictResponseItem:
    if mock_mode:
        raw = {
            "answer": "Uncertain" if "Uncertain" in req.options else (req.options[0] if req.options else "Uncertain"),
            "unit": "",
            "explanation": "MOCK_MODE response. Connect vLLM to generate real logic answers.",
            "premises_used": list(range(len(req.premises))),
            "reasoning": {"type": "mock", "steps": ["Mock response generated for API-format testing."]},
        }
        return coerce_response(req, raw)

    try:
        text = await llm.chat_json(SYSTEM_PROMPT_TYPE1, build_type1_prompt(req))
        raw = extract_json_object(text)
        return coerce_response(req, raw)
    except Exception as exc:
        return fallback_response(req, f"Type 1 pipeline failed safely: {exc}")

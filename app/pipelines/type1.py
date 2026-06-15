from __future__ import annotations

import json

from app.llm_client import VLLMClient
from app.schemas import PredictRequest, PredictResponseItem, fallback_response
from app.utils.json_utils import extract_json_object
from app.utils.normalize import coerce_response

SYSTEM_PROMPT_TYPE1 = """You are a rigorous logical reasoning AI expert specializing in First-Order Logic (FOL) for EXACT 2026.
Your task is to analyze the given natural language premises, construct formal FOL representations, deduce the logical conclusion, and identify the exact premises used.

Output format: You MUST return a single, valid JSON object with keys: "answer", "unit", "explanation", "premises_used", "reasoning". Do not wrap the JSON in markdown code blocks or add any text outside the JSON.

Strict Rules:
1. "answer" format:
   - If "options" is non-empty (choice questions), the value of "answer" MUST EXACTLY match one of the listed options (case-sensitive and character-perfect, e.g., "Yes", "No", "Uncertain").
2. "unit" format:
   - Must always be an empty string "" for Type 1 queries.
3. "premises_used" tracking:
   - Must contain a list of 0-based indices pointing to the input premises that are strictly necessary for the deduction (the first premise is index 0).
   - Double-check every index. Do not include premises that are irrelevant to the specific query.
   - If the answer is "Uncertain" due to a missing link, only list the premises that lead to the point of uncertainty (usually the premise stating the missing condition).
4. "explanation" format:
   - A concise, natural language explanation detailing step-by-step how the premises lead to the final answer.
5. "reasoning" format:
   - An object of shape {"type": "fol", "steps": [...]}.
   - In "steps", write down the formal FOL translations (using ForAll, Exists, Implies, And, Or, Not syntax) and the intermediate derivation steps.
"""

FEW_SHOT_CHOICE = """Below are examples of Choice Questions showing how to resolve logic deductions and select the correct supporting premises (0-based indices):

[Example 1: Multiple Choice Question (MCQ)]
Input Query:
{
  "query_id": "quick_type1_mc",
  "type": "type1",
  "query": "Based on the premises, which option is logically supported?\\nA. Asha may join Study Alpha\\nB. Asha cannot handle participant data\\nC. Asha has budget approval\\nD. Study Alpha has 20 enrolled participants",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": ["A", "B", "C", "D"]
}
Expected Output:
{
  "answer": "A",
  "unit": "",
  "explanation": "Since Asha completed ethics training (Premise 3) and has lab access (Premise 4), she can handle participant data (Premise 0). Since she can handle participant data and has supervisor approval (Premise 5), she may join Study Alpha (Premise 1). Therefore, Asha may join Study Alpha is supported.",
  "premises_used": [0, 1, 3, 4, 5],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Study_Alpha(x))",
      "completed_ethics_training(Asha)",
      "has_lab_access(Asha)",
      "has_supervisor_approval(Asha)",
      "may_join_Study_Alpha(Asha)"
    ]
  }
}

[Example 2: Yes/No/Uncertain Question (Yes Answer)]
Input Query:
{
  "query_id": "quick_type1_yes_no",
  "type": "type1",
  "query": "Is Asha listed as an active contributor?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": ["Yes", "No", "Uncertain"]
}
Expected Output:
{
  "answer": "Yes",
  "unit": "",
  "explanation": "Asha completed ethics training (Premise 3), has lab access (Premise 4), and has supervisor approval (Premise 5). Thus she can handle participant data (Premise 0) and may join Study Alpha (Premise 1). Since every researcher who may join Study Alpha is listed as an active contributor (Premise 2), Asha is listed as an active contributor.",
  "premises_used": [0, 1, 2, 3, 4, 5],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Study_Alpha(x))",
      "ForAll(x, (may_join_Study_Alpha(x) → listed_as_active_contributor(x)))",
      "completed_ethics_training(Asha)",
      "has_lab_access(Asha)",
      "has_supervisor_approval(Asha)",
      "listed_as_active_contributor(Asha)"
    ]
  }
}

[Example 3: Yes/No/Uncertain Question (Uncertain Answer)]
Input Query:
{
  "query_id": "quick_type1_uncertain",
  "type": "type1",
  "query": "Does Asha have budget approval?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": ["Yes", "No", "Uncertain"]
}
Expected Output:
{
  "answer": "Uncertain",
  "unit": "",
  "explanation": "No premise states whether Asha has budget approval (Premise 7). Thus, it is uncertain.",
  "premises_used": [7],
  "reasoning": {
    "type": "fol",
    "steps": [
      "No premise states whether Asha has budget approval"
    ]
  }
}
"""

FEW_SHOT_FREEFORM = """Below are examples of Free-form Questions showing how to resolve logic deductions and output short answers:

[Example 1: Free-form Numeric Question]
Input Query:
{
  "query_id": "quick_type1_number",
  "type": "type1",
  "query": "How many enrolled participants does Study Alpha have?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": []
}
Expected Output:
{
  "answer": "12",
  "unit": "",
  "explanation": "Study Alpha has 12 enrolled participants as explicitly stated in Premise 6.",
  "premises_used": [6],
  "reasoning": {
    "type": "fol",
    "steps": [
      "Study Alpha has 12 enrolled participants"
    ]
  }
}

[Example 2: Free-form Text Question]
Input Query:
{
  "query_id": "quick_type1_text",
  "type": "type1",
  "query": "Which researcher may join Study Alpha?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": []
}
Expected Output:
{
  "answer": "Asha",
  "unit": "",
  "explanation": "Since Asha completed ethics training (Premise 3) and has lab access (Premise 4), she can handle participant data (Premise 0). Since she can handle participant data and has supervisor approval (Premise 5), she may join Study Alpha (Premise 1).",
  "premises_used": [0, 1, 3, 4, 5],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Study_Alpha(x))",
      "completed_ethics_training(Asha)",
      "has_lab_access(Asha)",
      "has_supervisor_approval(Asha)",
      "may_join_Study_Alpha(Asha)"
    ]
  }
}
"""


def build_type1_prompt(req: PredictRequest) -> str:
    few_shot_example = FEW_SHOT_CHOICE if req.options else FEW_SHOT_FREEFORM
    return json.dumps(
        {
            "example_reference": few_shot_example,
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
        text = await llm.chat_json(SYSTEM_PROMPT_TYPE1, build_type1_prompt(req), query_id=req.query_id, pipeline="type1")
        raw = extract_json_object(text)
        return coerce_response(req, raw)
    except Exception as exc:
        return fallback_response(req, f"Type 1 pipeline failed safely: {exc}")

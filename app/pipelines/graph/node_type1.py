from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from app.pipelines.graph.state import AgentState
from app.schemas import PredictRequest
from app.utils.json_utils import extract_json_object

SYSTEM_PROMPT_TYPE1 = """You are a rigorous logical reasoning AI expert specializing in First-Order Logic (FOL) for EXACT 2026.
Your task is to analyze the given natural language premises, construct formal FOL representations, deduce the logical conclusion, identify the exact premises used, and write python code using z3-solver to solve/verify the logic problem.

Output format: You MUST return a single, valid JSON object with keys: "answer", "unit", "explanation", "premises_used", "reasoning", "z3_code". Do not wrap the JSON in markdown code blocks or add any text outside the JSON.

Strict Rules:
1. "answer" format:
   - If "options" is non-empty (choice questions), the value of "answer" MUST EXACTLY match one of the listed options (case-sensitive and character-perfect, e.g., "Yes", "No", "Uncertain").
2. "unit" format:
   - Must always be an empty string "" for Type 1 queries.
3. "premises_used" tracking:
   - Must contain a list of 0-based indices pointing to the input premises that are logically necessary and sufficient to deduce the final answer (the first premise is index 0).
   - ONLY include the minimal set of premises that participate directly in the reasoning chain leading to the answer.
   - Do NOT include redundant premises, rules that were not activated/triggered, or rules that just define entities/context without participating in the deduction path. Over-selecting premises is heavily penalized.
   - If the answer is "Uncertain" due to a missing link, list only the premises that lead up to the point of uncertainty (the rules that are actually active/relevant to the partial deduction) and the premise stating the missing condition if applicable.
4. "explanation" format:
   - A concise, natural language explanation detailing step-by-step how the premises lead to the final answer.
5. "reasoning" format:
   - An object of shape {"type": "fol", "steps": [...]}.
   - In "steps", write down the formal FOL translations (using ForAll, Exists, Implies, And, Or, Not syntax) and the intermediate derivation steps.
6. "z3_code" format:
   - A standalone Python script string that uses z3-solver (`from z3 import *`) to model the logic and print the final answer to stdout.
   - For choice questions (e.g. Yes/No/Uncertain), the script MUST print exactly one of the options (e.g., "Yes", "No", "Uncertain").
   - Z3 Variable Names constraint: Every Z3 variable/constant name MUST be a valid Python identifier, ASCII-only, with no spaces, hyphens (-), or special characters (replace them with underscores `_`).
   - Strict Closed-World Assumption (CWA) Modeling: A factual premise is only true if explicitly stated in the premises. If a fact or condition (e.g. `logs_reviewed`, `passes_linting`) is NOT explicitly stated as a fact in the premises, it is UNKNOWN (neither True nor False). In the Z3 code, you must DECLARE the corresponding Bool variable (e.g., `logs_reviewed = Bool('logs_reviewed')`) but you MUST NOT add any constraints on its value (do NOT call `s.add(logs_reviewed == True)` or `s.add(logs_reviewed == False)`). This allows Z3 to correctly yield unsatisfiable for negations only when it is actually proven under the given facts, returning 'Uncertain' otherwise.
   - To verify a logic query `Q` using Z3:
     * Check if premises imply `Q` by adding `Not(Q)` to the solver. If `check() == unsat`, it is proven ("Yes" or the matching option).
     * Else, check if premises imply `Not(Q)` by adding `Q` to the solver. If `check() == unsat`, it is disproven ("No" or contradiction).
     * Else, if both are satisfiable, it is "Uncertain".
     This ensures mathematical rigor and prevents over-inference.
   - For multiple choice questions (e.g. A, B, C, D), the script MUST check each option sequentially (e.g. by adding `Not(Option)` inside a push/pop block and checking if it is UNSAT) and print the option letter that is proven (e.g., print "A", "B", etc.). If no option is proven, print "Uncertain" or the default fallback option from the options.
7. "answer" and "z3_code" for math/arithmetic queries:
   - If the query asks for a mathematical calculation (e.g., "How many...", "What is the total..."), calculate the final value based strictly on the provided numeric facts. Do NOT output "Uncertain" or select "Uncertain" if the arithmetic calculation is fully determined by the premises.

Strict Anti-Hallucination Rules:
- Do NOT make assumptions that are not logically entailed. If a property or relationship is not specified in the premises, it must be considered "Uncertain" (neither True nor False). Under-specified conditions lead to "Uncertain" answers, NOT "No" or "Yes".
- Avoid the fallacy of denying the antecedent: If a premise states "If A then B", and you know "Not A", you CANNOT conclude "Not B". The correct answer in this case is "Uncertain" (unless other premises prove "Not B").
- Be extremely cautious of transitive relations. Only apply transitive deduction if a premise explicitly establishes a transitive link.
"""

FEW_SHOT_CHOICE = """Below are examples of Choice Questions showing how to resolve logic deductions, write Z3 solver code, and select the correct supporting premises (0-based indices):

[Example 1: Multiple Choice Question (MCQ)]
Input Query:
{
  "query_id": "quick_type1_mc",
  "type": "type1",
  "query": "Based on the premises, which option is logically supported?\\nA. John may join Project X\\nB. John cannot handle participant data\\nC. John has budget approval\\nD. Project X has 20 enrolled participants",
  "premises": [
    "[0] If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "[1] If a researcher can handle participant data and has supervisor approval, then that researcher may join Project X.",
    "[2] Every researcher who may join Project X is listed as an active contributor.",
    "[3] John completed ethics training.",
    "[4] John has lab access.",
    "[5] John has supervisor approval.",
    "[6] Project X has 12 enrolled participants.",
    "[7] No premise states whether John has budget approval."
  ],
  "options": ["A", "B", "C", "D"]
}
Expected Output:
{
  "answer": "A",
  "unit": "",
  "explanation": "Since John completed ethics training (Premise 3) and has lab access (Premise 4), she can handle participant data (Premise 0). Since she can handle participant data and has supervisor approval (Premise 5), she may join Project X (Premise 1). Therefore, John may join Project X is supported.",
  "premises_used": [0, 1, 3, 4, 5],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Project_X(x))",
      "completed_ethics_training(John)",
      "has_lab_access(John)",
      "has_supervisor_approval(John)",
      "may_join_Project_X(John)"
    ]
  },
  "z3_code": "import sys\\nfrom z3 import *\\ns = Solver()\\ncompleted_ethics_training_John = Bool('completed_ethics_training_John')\\nhas_lab_access_John = Bool('has_lab_access_John')\\ncan_handle_participant_data_John = Bool('can_handle_participant_data_John')\\nhas_supervisor_approval_John = Bool('has_supervisor_approval_John')\\nmay_join_Project_X_John = Bool('may_join_Project_X_John')\\nhas_budget_approval_John = Bool('has_budget_approval_John')\\nenrolled_participants = Int('enrolled_participants')\\ns.add(Implies(And(completed_ethics_training_John, has_lab_access_John), can_handle_participant_data_John))\\ns.add(Implies(And(can_handle_participant_data_John, has_supervisor_approval_John), may_join_Project_X_John))\\ns.add(completed_ethics_training_John == True)\\ns.add(has_lab_access_John == True)\\ns.add(has_supervisor_approval_John == True)\\ns.add(enrolled_participants == 12)\\ns.push()\\ns.add(Not(may_join_Project_X_John))\\nif s.check() == unsat:\\n    print('A')\\n    sys.exit(0)\\ns.pop()\\ns.push()\\ns.add(can_handle_participant_data_John)\\nif s.check() == unsat:\\n    print('B')\\n    sys.exit(0)\\ns.pop()\\ns.push()\\ns.add(Not(has_budget_approval_John))\\nif s.check() == unsat:\\n    print('C')\\n    sys.exit(0)\\ns.pop()\\ns.push()\\ns.add(Not(enrolled_participants == 20))\\nif s.check() == unsat:\\n    print('D')\\n    sys.exit(0)\\ns.pop()\\nprint('Uncertain')"
}

[Example 2: Yes/No/Uncertain Question (Yes Answer)]
Input Query:
{
  "query_id": "quick_type1_yes_no",
  "type": "type1",
  "query": "Is John listed as an active contributor?",
  "premises": [
    "[0] If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "[1] If a researcher can handle participant data and has supervisor approval, then that researcher may join Project X.",
    "[2] Every researcher who may join Project X is listed as an active contributor.",
    "[3] John completed ethics training.",
    "[4] John has lab access.",
    "[5] John has supervisor approval.",
    "[6] Project X has 12 enrolled participants.",
    "[7] No premise states whether John has budget approval."
  ],
  "options": ["Yes", "No", "Uncertain"]
}
Expected Output:
{
  "answer": "Yes",
  "unit": "",
  "explanation": "John completed ethics training (Premise 3), has lab access (Premise 4), and has supervisor approval (Premise 5). Thus she can handle participant data (Premise 0) and may join Project X (Premise 1). Since every researcher who may join Project X is listed as an active contributor (Premise 2), John is listed as an active contributor.",
  "premises_used": [0, 1, 2, 3, 4, 5],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Project_X(x))",
      "ForAll(x, (may_join_Project_X(x) → listed_as_active_contributor(x)))",
      "completed_ethics_training(John)",
      "has_lab_access(John)",
      "has_supervisor_approval(John)",
      "listed_as_active_contributor(John)"
    ]
  },
  "z3_code": "from z3 import *\\ns = Solver()\\ncompleted_ethics_training_John = Bool('completed_ethics_training_John')\\nhas_lab_access_John = Bool('has_lab_access_John')\\ncan_handle_participant_data_John = Bool('can_handle_participant_data_John')\\nhas_supervisor_approval_John = Bool('has_supervisor_approval_John')\\nmay_join_Project_X_John = Bool('may_join_Project_X_John')\\nlisted_as_active_contributor_John = Bool('listed_as_active_contributor_John')\\ns.add(Implies(And(completed_ethics_training_John, has_lab_access_John), can_handle_participant_data_John))\\ns.add(Implies(And(can_handle_participant_data_John, has_supervisor_approval_John), may_join_Project_X_John))\\ns.add(Implies(may_join_Project_X_John, listed_as_active_contributor_John))\\ns.add(completed_ethics_training_John == True)\\ns.add(has_lab_access_John == True)\\ns.add(has_supervisor_approval_John == True)\\ns.push()\\ns.add(Not(listed_as_active_contributor_John))\\nr_yes = s.check()\\ns.pop()\\nif r_yes == unsat:\\n    print('Yes')\\nelse:\\n    s.push()\\n    s.add(listed_as_active_contributor_John)\\n    r_no = s.check()\\n    s.pop()\\n    if r_no == unsat:\\n        print('No')\\n    else:\\n        print('Uncertain')"
}

[Example 3: Yes/No/Uncertain Question (Uncertain due to Missing Fact)]
Input Query:
{
  "query_id": "quick_type1_missing_fact",
  "type": "type1",
  "query": "Is John listed as an active contributor?",
  "premises": [
    "[0] If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "[1] If a researcher can handle participant data and has supervisor approval, then that researcher may join Project X.",
    "[2] Every researcher who may join Project X is listed as an active contributor.",
    "[3] John completed ethics training.",
    "[4] John has lab access."
  ],
  "options": ["Yes", "No", "Uncertain"]
}
Expected Output:
{
  "answer": "Uncertain",
  "unit": "",
  "explanation": "John completed ethics training (Premise 3) and has lab access (Premise 4), so she can handle participant data (Premise 0). However, it is not stated whether John has supervisor approval, which is required to join Project X (Premise 1) and be listed as an active contributor (Premise 2). Thus, it is uncertain.",
  "premises_used": [0, 1, 2, 3, 4],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Project_X(x))",
      "ForAll(x, (may_join_Project_X(x) → listed_as_active_contributor(x)))",
      "completed_ethics_training(John)",
      "has_lab_access(John)",
      "can_handle_participant_data(John)",
      "Unknown: has_supervisor_approval(John)"
    ]
  },
  "z3_code": "from z3 import *\\ns = Solver()\\ncompleted_ethics_training_John = Bool('completed_ethics_training_John')\\nhas_lab_access_John = Bool('has_lab_access_John')\\ncan_handle_participant_data_John = Bool('can_handle_participant_data_John')\\nhas_supervisor_approval_John = Bool('has_supervisor_approval_John')\\nmay_join_Project_X_John = Bool('may_join_Project_X_John')\\nlisted_as_active_contributor_John = Bool('listed_as_active_contributor_John')\\ns.add(Implies(And(completed_ethics_training_John, has_lab_access_John), can_handle_participant_data_John))\\ns.add(Implies(And(can_handle_participant_data_John, has_supervisor_approval_John), may_join_Project_X_John))\\ns.add(Implies(may_join_Project_X_John, listed_as_active_contributor_John))\\ns.add(completed_ethics_training_John == True)\\ns.add(has_lab_access_John == True)\\n# Note: has_supervisor_approval_John is not added as True or False because it is not specified in the premises\\ns.push()\\ns.add(Not(listed_as_active_contributor_John))\\nr_yes = s.check()\\ns.pop()\\nif r_yes == unsat:\\n    print('Yes')\\nelse:\\n    s.push()\\n    s.add(listed_as_active_contributor_John)\\n    r_no = s.check()\\n    s.pop()\\n    if r_no == unsat:\\n        print('No')\\n    else:\\n        print('Uncertain')"
}
"""

FEW_SHOT_FREEFORM = """Below are examples of Free-form Questions showing how to resolve logic deductions, write Z3 solver code, and output short answers:

[Example 1: Free-form Numeric Question]
Input Query:
{
  "query_id": "quick_type1_number",
  "type": "type1",
  "query": "How many enrolled participants does Project X have?",
  "premises": [
    "[0] If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "[1] If a researcher can handle participant data and has supervisor approval, then that researcher may join Project X.",
    "[2] Every researcher who may join Project X is listed as an active contributor.",
    "[3] John completed ethics training.",
    "[4] John has lab access.",
    "[5] John has supervisor approval.",
    "[6] Project X has 12 enrolled participants.",
    "[7] No premise states whether John has budget approval."
  ],
  "options": []
}
Expected Output:
{
  "answer": "12",
  "unit": "",
  "explanation": "Project X has 12 enrolled participants as explicitly stated in Premise 6.",
  "premises_used": [6],
  "reasoning": {
    "type": "fol",
    "steps": [
      "Project X has 12 enrolled participants"
    ]
  },
  "z3_code": "from z3 import *\\ns = Solver()\\nenrolled_participants_Project_X = Int('enrolled_participants_Project_X')\\ns.add(enrolled_participants_Project_X == 12)\\nif s.check() == sat:\\n    m = s.model()\\n    print(m[enrolled_participants_Project_X])"
}

[Example 2: Free-form Text Question]
Input Query:
{
  "query_id": "quick_type1_text",
  "type": "type1",
  "query": "Which researcher may join Project X?",
  "premises": [
    "[0] If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "[1] If a researcher can handle participant data and has supervisor approval, then that researcher may join Project X.",
    "[2] Every researcher who may join Project X is listed as an active contributor.",
    "[3] John completed ethics training.",
    "[4] John has lab access.",
    "[5] John has supervisor approval.",
    "[6] Project X has 12 enrolled participants.",
    "[7] No premise states whether John has budget approval."
  ],
  "options": []
}
Expected Output:
{
  "answer": "John",
  "unit": "",
  "explanation": "Since John completed ethics training (Premise 3) and has lab access (Premise 4), she can handle participant data (Premise 0). Since she can handle participant data and has supervisor approval (Premise 5), she may join Project X (Premise 1).",
  "premises_used": [0, 1, 3, 4, 5],
  "reasoning": {
    "type": "fol",
    "steps": [
      "ForAll(x, (completed_ethics_training(x) ∧ has_lab_access(x)) → can_handle_participant_data(x))",
      "ForAll(x, (can_handle_participant_data(x) ∧ has_supervisor_approval(x)) → may_join_Project_X(x))",
      "completed_ethics_training(John)",
      "has_lab_access(John)",
      "has_supervisor_approval(John)",
      "may_join_Project_X(John)"
    ]
  },
  "z3_code": "from z3 import *\\ns = Solver()\\ncompleted_ethics_training_John = Bool('completed_ethics_training_John')\\nhas_lab_access_John = Bool('has_lab_access_John')\\ncan_handle_participant_data_John = Bool('can_handle_participant_data_John')\\nhas_supervisor_approval_John = Bool('has_supervisor_approval_John')\\nmay_join_Project_X_John = Bool('may_join_Project_X_John')\\ns.add(Implies(And(completed_ethics_training_John, has_lab_access_John), can_handle_participant_data_John))\\ns.add(Implies(And(can_handle_participant_data_John, has_supervisor_approval_John), may_join_Project_X_John))\\ns.add(completed_ethics_training_John == True)\\ns.add(has_lab_access_John == True)\\ns.add(has_supervisor_approval_John == True)\\ns.push()\\ns.add(Not(may_join_Project_X_John))\\nr_yes = s.check()\\ns.pop()\\nif r_yes == unsat:\\n    print('John')\\nelse:\\n    print('Uncertain')"
}
"""


def build_type1_prompt(req: PredictRequest) -> str:
    few_shot_example = FEW_SHOT_CHOICE if req.options else FEW_SHOT_FREEFORM
    indexed_premises = [f"[{i}] {premise}" for i, premise in enumerate(req.premises)]
    return json.dumps(
        {
            "example_reference": few_shot_example,
            "query_id": req.query_id,
            "type": req.type,
            "query": req.query,
            "premises": indexed_premises,
            "options": req.options,
            "required_output_shape": {
                "answer": "exact option if options non-empty, otherwise short answer",
                "unit": "",
                "explanation": "non-empty explanation",
                "premises_used": "list of 0-based premise indices used",
                "reasoning": {"type": "fol", "steps": ["step 1", "step 2"]},
                "z3_code": "standalone Python code using z3-solver that prints only the final answer",
            },
        },
        ensure_ascii=False,
    )


def build_type1_feedback_prompt(req: PredictRequest, prev_raw: dict[str, Any], feedback_message: str) -> str:
    few_shot_example = FEW_SHOT_CHOICE if req.options else FEW_SHOT_FREEFORM
    indexed_premises = [f"[{i}] {premise}" for i, premise in enumerate(req.premises)]
    return json.dumps(
        {
            "example_reference": few_shot_example,
            "query_id": req.query_id,
            "type": req.type,
            "query": req.query,
            "premises": indexed_premises,
            "options": req.options,
            "previous_attempt": prev_raw,
            "feedback": feedback_message,
            "instruction": "Your previous attempt had errors or inconsistencies as described in 'feedback'. "
            "Please review the premises, fix all syntax or logical errors in the Z3 code, "
            "ensure the direct 'answer' and 'z3_code' execution match perfectly, and regenerate the JSON.",
            "required_output_shape": {
                "answer": "exact option if options non-empty, otherwise short answer",
                "unit": "",
                "explanation": "non-empty explanation",
                "premises_used": "list of 0-based premise indices used",
                "reasoning": {"type": "fol", "steps": ["step 1", "step 2"]},
                "z3_code": "standalone Python code using z3-solver that prints only the final answer",
            },
        },
        ensure_ascii=False,
    )


async def llm_type1_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node that calls LLM Type 1 (logic) with Z3 code generation and self-correction."""
    llm_client = config.get("configurable", {}).get("type1_llm")
    if not llm_client:
        raise ValueError("type1_llm client must be provided in config['configurable']")

    req = PredictRequest(
        query_id=state["query_id"],
        type=state["qtype"],
        query=state["query"],
        premises=state["premises"],
        options=state["options"],
    )

    retry_count = state.get("retry_count", 0)
    attempts = state.get("attempts_history", [])

    if retry_count > 0 and attempts:
        latest_attempt = attempts[-1]
        prev_raw = {
            "answer": state.get("llm_answer", ""),
            "unit": state.get("unit", ""),
            "explanation": state.get("explanation", ""),
            "premises_used": state.get("premises_used", []),
            "reasoning": state.get("reasoning", {}),
            "z3_code": state.get("generated_code", ""),
        }
        feedback_msg = latest_attempt.get("feedback", "Contradiction or execution failure.")
        prompt_str = build_type1_feedback_prompt(req, prev_raw, feedback_msg)
        pipeline_name = f"type1_graph_retry_{retry_count}"
    else:
        prompt_str = build_type1_prompt(req)
        pipeline_name = "type1_graph_initial"

    try:
        text = await llm_client.chat_json(
            SYSTEM_PROMPT_TYPE1,
            prompt_str,
            query_id=state["query_id"],
            pipeline=pipeline_name,
            max_tokens=llm_client.settings.type1_llm_max_tokens,
        )
        raw = extract_json_object(text)
    except Exception as exc:
        return {
            "llm_answer": "Uncertain" if "Uncertain" in state["options"] else (state["options"][0] if state["options"] else "Uncertain"),
            "explanation": f"LLM Type 1 call failed: {exc}",
            "generated_code": "",
        }

    return {
        "llm_answer": str(raw.get("answer", "")).strip(),
        "unit": "",
        "explanation": str(raw.get("explanation", "")).strip(),
        "premises_used": raw.get("premises_used", []),
        "reasoning": raw.get("reasoning", {}),
        "generated_code": str(raw.get("z3_code", "")).strip(),
    }

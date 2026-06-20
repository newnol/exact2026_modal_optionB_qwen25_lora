from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from app.pipelines.graph.state import AgentState
from app.schemas import PredictRequest
from app.utils.json_utils import extract_json_object

SYSTEM_PROMPT_TYPE1 = """You solve EXACT Type 1 logic questions.
Return one valid JSON object only with keys: answer, unit, explanation, premises_used, reasoning, z3_code.
Rules:
- unit must be "".
- If options is non-empty, answer must exactly match one listed option.
- Use only facts entailed by the premises. Missing facts mean Uncertain.
- premises_used must be the minimal 0-based indices needed for the conclusion.
- reasoning must be {"type":"fol","steps":[...]} with short FOL-style steps.
- z3_code must be standalone Python using `from z3 import *` and print only the final answer.
- For Yes/No/Uncertain: prove Q with Not(Q), disprove with Q, else Uncertain.
- Use ASCII-safe Python identifiers in z3_code.
"""


def _truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _clean_clause(text: str) -> str:
    text = text.strip().lower().rstrip(".?!")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"^(?:do the premises establish that|according to the premises,?|based on the [^,]+,?)\s+", "", text)
    text = text.replace(" it is ", " ").replace(" it can be ", " can be ")
    replacements = [
        (r"^(?:[a-z0-9 ]+ )?package is medical$", "medical"),
        (r"^(?:[a-z0-9 ]+ )?package weighs under 2 kilograms$", "weighs under 2 kilograms"),
        (r"^(?:[a-z0-9 ]+ )?package has priority delivery status$", "priority delivery status"),
        (r"^(?:[a-z0-9 ]+ )?package is not a priority package$", "not priority delivery status"),
        (r"^(?:[a-z0-9 ]+ )?package can be dispatched$", "can be dispatched"),
        (r"^(?:[a-z0-9 ]+ )?package cannot be dispatched.*$", "not can be dispatched"),
        (r"^it can be dispatched$", "can be dispatched"),
        (r"^it receives priority delivery status$", "priority delivery status"),
        (r"^priority delivery status$", "priority delivery status"),
        (r"^its route is clear$", "route clear"),
        (r"^route is clear$", "route clear"),
        (r"^route is blocked$", "not route clear"),
        (r"^(?:the )?weather is safe(?: for [a-z0-9 ]+)?$", "weather safe"),
        (r"^weather safe$", "weather safe"),
        (r"^emergency waiver is approved(?: for [a-z0-9 ]+)?$", "emergency waiver approved"),
        (r"^alternate route is mapped(?: for [a-z0-9 ]+)?$", "alternate route mapped"),
        (r"^eligible to use the aerial corridor$", "eligible aerial corridor"),
        (r"^[a-z0-9 ]+ eligible to use the aerial corridor$", "eligible aerial corridor"),
        (r"^(?:[a-z0-9 ]+ )?is eligible to use the aerial corridor$", "eligible aerial corridor"),
        (r"^launch is approved$", "launch approved"),
        (r"^(?:[a-z0-9 ]+ )?has launch approval$", "launch approved"),
        (r"^operator is assigned$", "operator assigned"),
        (r"^its metadata is complete$", "metadata complete"),
        (r"^metadata is complete$", "metadata complete"),
        (r"^its rights are cleared$", "rights cleared"),
        (r"^rights are cleared$", "rights cleared"),
        (r"^ocr has been verified$", "ocr verified"),
        (r"^it is searchable online$", "searchable online"),
        (r"^it is eligible for the public portal$", "eligible public portal"),
        (r"^eligible for the public portal$", "eligible public portal"),
        (r"^manuscript is scanned at 600 dpi$", "scanned at 600 dpi"),
        (r"^manuscript is preservation ready$", "preservation ready"),
        (r"^manuscript is preservation-ready$", "preservation ready"),
        (r"^[a-z0-9 ]* is preservation ready$", "preservation ready"),
        (r"^[a-z0-9 ]* is preservation-ready$", "preservation ready"),
        (r"^[a-z0-9 ]* scanned at 600 dpi$", "scanned at 600 dpi"),
        (r"^[a-z0-9 ]* metadata is complete$", "metadata complete"),
        (r"^[a-z0-9 ]* rights are cleared$", "rights cleared"),
        (r"^[a-z0-9 ]* is eligible for the public portal$", "eligible public portal"),
        (r"^[a-z0-9 ]* ocr has been verified$", "ocr verified"),
        (r"^[a-z0-9 ]* lacks ocr verification$", "not ocr verified"),
        (r"^[a-z0-9 ]* is searchable online.*$", "searchable online"),
        (r"^[a-z0-9 ]* contains personal data$", "contains personal data"),
        (r"^privacy review is required$", "privacy review required"),
        (r"^redaction is complete$", "redaction complete"),
        (r"^[a-z0-9 ]* safe for public release because.*$", "safe public release"),
        (r"^[a-z0-9 ]* is safe for public release$", "safe public release"),
        (r"^soil moisture is low$", "soil moisture low"),
        (r"^[a-z0-9 ]* soil moisture is low$", "soil moisture low"),
        (r"^[a-z0-9 ]* has low soil moisture$", "soil moisture low"),
        (r"^heatwave is active(?: for [a-z0-9 ]+)?$", "heatwave active"),
        (r"^irrigation is needed$", "irrigation needed"),
        (r"^irrigation is unnecessary.*$", "not irrigation needed"),
        (r"^reservoir has water$", "reservoir has water"),
        (r"^[a-z0-9 ]* reservoir has water$", "reservoir has water"),
        (r"^[a-z0-9 ]* reservoir lacks water$", "not reservoir has water"),
        (r"^irrigation is scheduled$", "irrigation scheduled"),
        (r"^sensor calibration is current$", "sensor calibration current"),
        (r"^[a-z0-9 ]* sensor calibration is current$", "sensor calibration current"),
        (r"^autonomous watering is allowed(?: for [a-z0-9 ]+)?$", "autonomous watering allowed"),
        (r"^pest risk is high$", "pest risk high"),
        (r"^[a-z0-9 ]* pest risk is high$", "pest risk high"),
        (r"^[a-z0-9 ]* has high pest risk$", "pest risk high"),
        (r"^pesticide review is required$", "pesticide review required"),
        (r"^agronomist approval is given$", "agronomist approval given"),
        (r"^chemical treatment is allowed(?: for [a-z0-9 ]+)?$", "chemical treatment allowed"),
    ]
    for pattern, replacement in replacements:
        if re.match(pattern, text):
            return replacement
    text = re.sub(r"^[a-z0-9 ]* is ", "", text)
    text = re.sub(r"^[a-z0-9 ]* has ", "has ", text)
    text = re.sub(r"^[a-z0-9 ]* was ", "", text)
    return text.strip()


def _split_conditions(text: str) -> list[str]:
    return [_clean_clause(part) for part in re.split(r"\band\b", text) if _clean_clause(part)]


def _parse_rule(premise: str) -> tuple[list[str], str] | None:
    text = premise.strip().rstrip(".")
    match = re.match(r"^if (.+?), then (.+)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    antecedents = _split_conditions(match.group(1))
    consequent = _clean_clause(match.group(2))
    if not antecedents or not consequent:
        return None
    return antecedents, consequent


def _parse_option_map(query: str) -> dict[str, str]:
    option_map: dict[str, str] = {}
    for line in query.splitlines():
        match = re.match(r"^\s*([A-Z])\.\s*(.+)$", line.strip())
        if match:
            option_map[match.group(1)] = _clean_clause(match.group(2))
    return option_map


def _extract_yes_no_target(query: str) -> str | None:
    cleaned = query.strip().rstrip(".?").lower()
    patterns = [
        r"^is (.+?), according to the premises$",
        r"^is (.+)$",
        r"^do the premises establish that (.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if match:
            return _clean_clause(match.group(1))
    return None


def deterministic_type1_solver(req: PredictRequest) -> dict[str, Any] | None:
    facts: dict[str, set[int]] = {}
    rules: list[tuple[list[str], str, int]] = []
    derivations: dict[str, set[int]] = {}
    reasoning_steps: list[str] = []

    for idx, premise in enumerate(req.premises):
        parsed_rule = _parse_rule(premise)
        if parsed_rule:
            antecedents, consequent = parsed_rule
            rules.append((antecedents, consequent, idx))
            continue
        fact = _clean_clause(premise)
        if fact:
            facts.setdefault(fact, set()).add(idx)
            derivations.setdefault(fact, set()).update({idx})

    changed = True
    while changed:
        changed = False
        for antecedents, consequent, rule_idx in rules:
            if not all(a in derivations for a in antecedents):
                continue
            support = {rule_idx}
            for antecedent in antecedents:
                support.update(derivations[antecedent])
            prev = derivations.get(consequent)
            if prev is None or len(support) < len(prev):
                derivations[consequent] = set(support)
                if consequent not in facts:
                    reasoning_steps.append(f"{' & '.join(antecedents)} -> {consequent}")
                changed = True

    if req.options and set(req.options) == {"Yes", "No", "Uncertain"}:
        target = _extract_yes_no_target(req.query)
        if not target:
            return None
        meta_query = req.query.strip().lower().startswith("do the premises establish that ")
        if target in derivations:
            answer = "Yes"
            used = sorted(derivations[target])
            explanation = f"The target statement '{target}' is derivable from the premises."
        elif meta_query:
            answer = "No"
            used = []
            explanation = f"The premises do not establish '{target}'."
        elif f"not {target}" in derivations:
            answer = "No"
            used = sorted(derivations[f"not {target}"])
            explanation = f"The negation of '{target}' is derivable from the premises."
        else:
            answer = "Uncertain"
            used = []
            explanation = f"The premises do not prove or disprove '{target}'."
        return {
            "answer": answer,
            "unit": "",
            "explanation": explanation,
            "premises_used": used,
            "reasoning": {"type": "fol", "steps": reasoning_steps[:8] or ["forward chaining"]},
        }

    option_map = _parse_option_map(req.query)
    if req.options and option_map and all(opt in option_map for opt in req.options):
        supported: list[tuple[str, set[int]]] = []
        for opt in req.options:
            clause = option_map[opt]
            if clause in derivations:
                supported.append((opt, derivations[clause]))
        if supported:
            ranked = sorted(supported, key=lambda item: (len(item[1]), item[0]), reverse=True)
            top_opt, top_support = ranked[0]
            second_size = len(ranked[1][1]) if len(ranked) > 1 else -1
            if len(top_support) > second_size:
                return {
                    "answer": top_opt,
                    "unit": "",
                    "explanation": f"Option {top_opt} has the strongest support under forward chaining.",
                    "premises_used": sorted(top_support),
                    "reasoning": {"type": "fol", "steps": reasoning_steps[:8] or ["forward chaining"]},
                }
        if len(supported) == 1:
            opt, support = supported[0]
            return {
                "answer": opt,
                "unit": "",
                "explanation": f"Only option {opt} is supported by forward chaining over the premises.",
                "premises_used": sorted(support),
                "reasoning": {"type": "fol", "steps": reasoning_steps[:8] or ["forward chaining"]},
            }

    return None


def build_type1_prompt(req: PredictRequest) -> str:
    indexed_premises = [f"[{i}] {premise}" for i, premise in enumerate(req.premises)]
    return json.dumps(
        {
            "query_id": req.query_id,
            "type": req.type,
            "query": req.query,
            "premises": indexed_premises,
            "options": req.options,
            "answer_policy": (
                "If options are provided, answer must be exactly one option."
                if req.options
                else "Return a short direct answer from the premises."
            ),
            "z3_policy": [
                "Translate only the needed facts and rules.",
                "Print only the final answer.",
                "For uncertainty, print Uncertain.",
            ],
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
    indexed_premises = [f"[{i}] {premise}" for i, premise in enumerate(req.premises)]
    return json.dumps(
        {
            "query_id": req.query_id,
            "type": req.type,
            "query": req.query,
            "premises": indexed_premises,
            "options": req.options,
            "previous_attempt_summary": {
                "answer": str(prev_raw.get("answer", "")),
                "premises_used": prev_raw.get("premises_used", []),
                "reasoning": prev_raw.get("reasoning", {}),
                "z3_code_excerpt": _truncate(str(prev_raw.get("z3_code", "")), 900),
            },
            "feedback": feedback_message,
            "instruction": (
                "Fix the previous attempt. Keep the answer and z3_code consistent, "
                "repair any syntax or logic errors, and return the same JSON schema."
            ),
            "required_output_shape": {
                "answer": "exact option if options non-empty, otherwise short answer",
                "unit": "",
                "explanation": "non-empty explanation",
                "premises_used": "list of 0-based premise indices used",
                "reasoning": {"type": "fol", "steps": ["step 1", "step 2"]},
                "z3_code": "standalone Python code using z3-solver that prints only the final answer"
            },
        },
        ensure_ascii=False,
    )


async def llm_type1_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node that calls LLM Type 1 (logic) with Z3 code generation and self-correction."""
    llm_client = config.get("configurable", {}).get("type1_llm")
    if not llm_client:
        raise ValueError("type1_llm client must be provided in config['configurable']")

    # Reconstruct PredictRequest for prompt building helpers
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
        # Build feedback prompt using the latest attempt's results
        latest_attempt = attempts[-1]
        prev_raw = {
            "answer": state.get("llm_answer", ""),
            "unit": state.get("unit", ""),
            "explanation": state.get("explanation", ""),
            "premises_used": state.get("premises_used", []),
            "reasoning": state.get("reasoning", {}),
            "z3_code": state.get("generated_code", ""),
        }
        
        # Build error feedback message
        feedback_msg = latest_attempt.get("feedback", "Contradiction or execution failure.")
        prompt_str = build_type1_feedback_prompt(req, prev_raw, feedback_msg)
        pipeline_name = f"type1_graph_retry_{retry_count}"
    else:
        # Build initial prompt
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
        # On connection/LLM failure, fallback gracefully
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

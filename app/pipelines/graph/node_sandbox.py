from __future__ import annotations

from typing import Any, Dict

from app.pipelines.graph.state import AgentState
from app.utils.sandbox import run_python_code


def sandbox_node(state: AgentState) -> Dict[str, Any]:
    """Sandbox execution node that runs the generated python code (Z3 or SymPy)."""
    code = state.get("generated_code", "").strip()
    if not code:
        return {
            "sandbox_ok": False,
            "sandbox_output": "",
            "sandbox_error": "No generated code found in state",
        }

    # Execute code in sandbox
    result = run_python_code(code, timeout_seconds=3.0)

    return {
        "sandbox_ok": result.ok,
        "sandbox_output": result.stdout,
        "sandbox_error": result.stderr if not result.ok else "",
    }

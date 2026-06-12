from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def run_python_code(code: str, timeout_seconds: float = 3.0) -> SandboxResult:
    """Run model-generated Python in a short-lived subprocess.

    This is a lightweight inference-time sandbox for competition physics solvers.
    It is not a security boundary for untrusted public users; keep the endpoint private-ish
    to the evaluator and avoid exposing arbitrary code execution to the open internet.
    """
    code = (code or "").strip()
    if not code:
        return SandboxResult(False, "", "empty python_code", 1)

    prelude = """
import math
try:
    import sympy as sp
except Exception:
    sp = None
""".strip()

    full_code = prelude + "\n\n" + code + "\n"
    with tempfile.TemporaryDirectory(prefix="exact_sandbox_") as td:
        script = os.path.join(td, "solve.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(full_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", script],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
            )
            return SandboxResult(
                ok=proc.returncode == 0 and bool(proc.stdout.strip()),
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(False, exc.stdout or "", exc.stderr or "timeout", 124, timed_out=True)


def last_stdout_value(stdout: str) -> str:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""

# Project Architecture

## Repository goal

This repository deploys an EXACT 2026-compatible `POST /predict` endpoint and a `GET /v1/models` endpoint using:

- shared base model: `Qwen/Qwen2.5-7B-Instruct`
- Type 1 LoRA: `type1-logic`
- Type 2 LoRA: `type2-physics`

The project is built around one shared vLLM server with both LoRAs loaded at startup. The API wrapper selects the already-loaded adapter by choosing the OpenAI-compatible `model` field.

## Main runtime surface

### API

- `app/main.py`
  - `POST /predict`
  - `GET /healthz`
- vLLM `GET /v1/models`
  - exposed by the Modal deployment in `modal_exact2026.py`

`/predict` always returns a list with one item, which matches the competition schema.

### Request schema

Defined in `app/schemas.py`.

Input:

- `query_id`
- `type`: `type1` or `type2`
- `query`
- `premises`
- `options`

Output:

- `query_id`
- `answer`
- `unit`
- `explanation`
- `premises_used`
- `reasoning`

There is also a fallback response path that preserves schema validity when model execution fails.

## LangGraph flow

The graph is assembled in `app/pipelines/graph/__init__.py`.

### Router

`node_router.py` chooses among:

- `type1`
- `type2`
- `fast_path`

For Type 2, if the query matches a small deterministic pattern, the request goes straight to a conservative solver without LLM code generation.

### Type 1 path

Files:

- `node_type1.py`
- `node_sandbox.py`
- `node_check_type1.py`
- `node_formatter.py`

Flow:

1. Prompt the Type 1 LoRA to return:
   - final answer
   - explanation
   - `premises_used`
   - FOL-style reasoning
   - generated `z3_code`
2. Run generated code in a short-lived Python subprocess.
3. Compare LLM answer against sandbox output.
4. Retry up to two additional times when code fails or answer/code disagree.
5. Normalize the final response before returning it.

Important design point:

- Type 1 tries to use executable Z3 code as a consistency check, not just free-form text reasoning.

### Type 2 path

Files:

- `node_type2.py`
- `node_sandbox.py`
- `node_check_type2.py`
- `node_formatter.py`

Flow:

1. Try a deterministic fast path for common public-style parallel resistor questions.
2. Otherwise prompt the Type 2 LoRA to generate standalone Python/SymPy code.
3. Execute the generated code in the sandbox.
4. If code execution fails or prints nothing, fall back to a direct-answer LLM prompt.
5. Normalize the unit to ASCII-safe output before returning it.

Important design point:

- Type 2 is optimized for numeric answers and unit separation.

## Output normalization

`app/utils/normalize.py` does the last-mile cleanup:

- forces Type 1 multiple-choice answers to exactly match an allowed option
- defaults bad Type 1 outputs toward a valid answer string
- normalizes Type 2 units into ASCII forms such as `ohm` and `uF`
- audits and adjusts `premises_used` using explanation and reasoning text

That premise audit matters because the competition scores Type 1 supporting premises explicitly.

## Sandbox behavior

`app/utils/sandbox.py` runs model-generated Python with:

- isolated interpreter mode
- short timeout
- no file/network access through the generated code path
- captured stdout used as the executable answer

This is an inference-time guardrail, not a hardened public security boundary.

## Deployment model

`modal_exact2026.py` defines two Modal services:

1. `vllm_server`
   - serves `Qwen/Qwen2.5-7B-Instruct`
   - loads both LoRAs at startup
2. `predict_api`
   - FastAPI wrapper
   - points Type 1 and Type 2 requests at the shared vLLM backend

Relevant environment behavior:

- `TYPE1_MODEL_NAME=type1-logic`
- `TYPE2_MODEL_NAME=type2-physics`
- `REQUEST_TIMEOUT_SECONDS=50`
- `MOCK_MODE=false` in deployed mode

## Logging and observability

The repo logs structured JSONL events via `app/logging_utils.py` and the main request path.

Logged event types described in `README.md`:

- `predict`
- `llm_call`
- `premise_audit`

This is useful for public test day preparation and latency review.

## Tests and utilities

Core files:

- `tests/test_api_shape.py`
- `tests/test_langgraph_pipeline.py`
- `tests/test_endpoint.py`
- `tests/test_suites.py`

Utility scripts:

- `scripts/preflight_check_loras.py`
- `scripts/test_full.sh`
- `scripts/modal_test_public.sh`
- `scripts/modal_latency_check.py`
- `scripts/build_submission_package.sh`
- `scripts/deploy.sh`

## Submission flow supported by the repo

1. Validate LoRA base compatibility.
2. Deploy Modal services.
3. Capture the `/predict` and `/v1/models` URLs.
4. Run public endpoint checks.
5. Add `submission/solution.pdf`.
6. Build the final zip package.

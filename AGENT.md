# AGENT.md

This file records the project state and the decisions that matter so a later agent can continue work without rediscovering the same issues.

## Goal

Build and operate an EXACT 2026 Option B submission around:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- Type 1 LoRA: `type1-logic`
- Type 2 LoRA: `type2-physics`
- one shared vLLM server on Modal
- one `/predict` API wrapper

## Current architecture

### Deploy

- Deploy file: [modal_exact2026.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/modal_exact2026.py)
- Modal app name: `exact2026-optionb-qwen25`
- vLLM serves:
  - base model
  - 2 LoRA adapters loaded at startup
- `/predict` wrapper selects adapter by OpenAI `model` field

### Important deploy settings

- vLLM `--max-model-len` is now `8192`
- vLLM metrics middleware is patched by:
  - [scripts/patch_vllm_metrics.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/scripts/patch_vllm_metrics.py)
- This patch exists because `prometheus_fastapi_instrumentator` crashed vLLM routes with:
  - `_IncludedRouter` has no attribute `path`

### Type 1 pipeline

Type 1 is now intentionally `LLM-first`:

1. `llm_type1`
2. `sandbox`
3. `check_type1`
4. retry if needed
5. `formatter`

Deterministic Type 1 logic still exists, but only as a fallback/verifier path inside formatter when:

- retries are exhausted
- there is still no verified answer from sandbox

Relevant files:

- [app/pipelines/graph/node_type1.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/pipelines/graph/node_type1.py)
- [app/pipelines/graph/node_check_type1.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/pipelines/graph/node_check_type1.py)
- [app/pipelines/graph/node_formatter.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/pipelines/graph/node_formatter.py)
- [app/pipelines/graph/node_router.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/pipelines/graph/node_router.py)
- [app/pipelines/graph/__init__.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/pipelines/graph/__init__.py)

### Type 2 pipeline

Type 2 is still hybrid:

1. deterministic physics fast path if pattern matches
2. otherwise code-generation LLM path
3. sandbox execution
4. direct LLM fallback if code path fails

Relevant file:

- [app/pipelines/graph/node_type2.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/pipelines/graph/node_type2.py)

## Important runtime behavior

### Modal rollout

`modal deploy` alone is not always enough to replace currently serving containers.

After deploy, run:

```bash
.venv/bin/modal app rollover exact2026-optionb-qwen25
```

Without `rollover`, the endpoint may keep serving old behavior.

### Python environment

Project local env:

- `.venv`
- installed with `uv`

Useful commands:

```bash
.venv/bin/modal deploy modal_exact2026.py
.venv/bin/modal app rollover exact2026-optionb-qwen25
.venv/bin/python tests/test_regression_suite.py
```

## Errors already diagnosed and fixed

### 1. vLLM 500 on `/v1/models` and `/v1/chat/completions`

Root cause:

- `prometheus_fastapi_instrumentator`
- route inspection crash inside vLLM metrics setup

Fix:

- patch out the Instrumentator wiring
- keep `/metrics` lightweight

### 2. Type 1 `400 Bad Request` because context overflow

Root cause:

- huge Type 1 prompt
- `max_tokens=768`
- vLLM context previously capped at `4096`

Fixes made:

- shortened Type 1 prompts
- added type-specific token override support in:
  - [app/llm_client.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/llm_client.py)
  - [app/config.py](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/app/config.py)
- increased vLLM `--max-model-len` to `8192`

## Current test status

### Regression suite

Latest full regression snapshot:

- file: [test_results/datatest_regression_1781924850.txt](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/test_results/datatest_regression_1781924850.txt)
- result: `21/23`

Remaining regression failures there:

- `REG_T2_0018`
- `REG_T2_GEN_022`

### datatest2

Latest `datatest2` snapshot:

- file: [test_results/datatest2_1781925728.txt](/Users/newnol/workspace/03-competition/exact2026_optionb_langgraph/test_results/datatest2_1781925728.txt)
- result: `39/50 = 78.0%`
- Type1: `17/25`
- Type2: `22/25`

Current `datatest2` failures:

- Type1:
  - `T1_0033`
  - `T1_0046`
  - `T1_0013`
  - `T1_0041`
  - `T1_0034`
  - `T1_0042`
  - `T1_0024`
  - `T1_0008`
- Type2:
  - `T2_0013`
  - `T2_0018`
  - `T2_0020`

## Key diagnosis

### Type 1

- correctness improved after switching to `LLM-first`
- but latency is still high on `datatest2`
- many Type 1 requests take around `30s`
- current remaining failures look like reasoning/sign errors, not infra errors

### Type 2

Most remaining issues are still engineering issues:

- exact formatting
- numeric normalization
- one or two missing/incorrect deterministic solver paths

This means training is not the next best move yet.

## Recommended next steps

Priority order:

1. Fix the 3 remaining Type 2 failures:
   - `T2_0013`
   - `T2_0018`
   - `T2_0020`
2. Open the 8 remaining Type 1 failures and group them by reasoning pattern
3. Reduce Type 1 latency if needed, but correctness is currently the higher-value issue
4. Only consider more fine-tuning after pipeline errors are mostly squeezed out

## Design stance

Do not make deterministic Type 1 the primary route.

Reason:

- user explicitly wants Type 1 to remain AI-driven
- deterministic logic can overfit visible test structure

So the intended design is:

- LLM does reasoning
- sandbox verifies
- deterministic fallback exists only as a last resort


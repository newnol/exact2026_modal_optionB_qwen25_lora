# Competition Fit and Current Risks

This document maps the current repository against the EXACT 2026 requirements checked on June 20, 2026.

## What already matches the competition well

### 1. API shape is aligned

The competition requires a JSON list response with:

- `query_id`
- `answer`
- `unit`
- `explanation`
- `premises_used`
- `reasoning`

This repo returns exactly that shape from `app/main.py` and `app/schemas.py`.

### 2. Model policy appears aligned

The current deployment plan uses:

- `Qwen/Qwen2.5-7B-Instruct`
- Type 1 LoRA on Qwen2.5
- Type 2 LoRA on Qwen2.5

That is consistent with the public rule that all LLMs must be open-source and 8B parameters or fewer.

### 3. Type 1 scoring target is directly reflected in the design

The official hidden-test round scores Type 1 on:

- answer correctness
- supporting-premise correctness

This repo explicitly optimizes for both:

- generated Z3 code for answer verification
- `premises_used` audit and normalization in `app/utils/normalize.py`

### 4. Type 2 scoring target is directly reflected in the design

The official hidden-test round scores Type 2 on:

- answer value
- unit

This repo directly targets those fields through:

- generated Python/SymPy computation
- deterministic fast path for common circuit questions
- ASCII unit normalization

### 5. Explanation is always preserved

The competition requires a non-empty explanation. The code normalizes missing explanations into a fallback string instead of returning an empty field.

## Where the repo is competition-aware in a practical way

### Shared vLLM plus two LoRAs

The project is intentionally built around one base model and two preloaded LoRAs. That reduces model-swap overhead and matches the need to stay responsive during grading.

### Timeout budgeting

`REQUEST_TIMEOUT_SECONDS=50` keeps the LLM call under the typical 60-second evaluator timeout budget described in the repo comments.

### Public-test preparation

The repo includes:

- warm-up scripts
- endpoint tests
- latency checks
- submission zip assembly

That is the right operational surface for the competition.

## Current risks and caveats

### 1. The hidden test does not reward Type 2 explanations directly

The official page says hidden-test automated scoring for Type 2 uses numerical answer plus unit, while explanations remain required only for schema completeness. This means effort spent on Type 2 explanation quality helps the final live round more than the hidden round.

Practical implication:

- during hidden-test optimization, numeric robustness and unit correctness should dominate

### 2. Type 1 premise auditing is heuristic

`app/utils/normalize.py` contains a post-hoc heuristic auditor for `premises_used`. That is useful, but it also means premise indices are not guaranteed to be faithful to the model's real derivation.

Practical implication:

- if leaderboard performance on Type 1 premises is weaker than expected, this file is the first place to inspect

### 3. Type 2 deterministic solver is intentionally narrow

The fast path currently recognizes a limited subset of problems. Most Type 2 performance still depends on generated code or direct-answer fallback.

Practical implication:

- expanding deterministic solvers for common formula families could materially improve hidden-test stability

### 4. Sandbox reliability is part of answer quality

For Type 1, the graph retries when Z3 code fails or disagrees with the textual answer. For Type 2, the graph falls back if generated code fails. This is robust, but it means prompt quality and executable code quality are tightly coupled to final accuracy.

Practical implication:

- log review of `llm_call`, `predict`, and sandbox failures is operationally important before live evaluation

### 5. The repo comments mention evaluator timeout assumptions, but the official page does not publish a per-request timeout

The code is built around a practical timeout budget, which is sensible. Still, that number comes from repo assumptions and deployment experience, not from the competition page itself.

Practical implication:

- keep latency measurements from the deployed Modal endpoint, especially before June 25, 2026 public test day

## Suggested reading of the repo for future changes

If the next task is improving competition performance, start in this order:

1. `app/utils/normalize.py`
2. `app/pipelines/graph/node_type1.py`
3. `app/pipelines/graph/node_type2.py`
4. `app/pipelines/graph/node_check_type1.py`
5. `modal_exact2026.py`

## Bottom line

This repository is structurally aligned with EXACT 2026:

- schema matches
- model policy appears compliant
- Type 1 path targets answer plus premise correctness
- Type 2 path targets numeric answer plus unit
- deployment and submission utilities are competition-oriented

The main technical risk is not competition mismatch. It is runtime accuracy and stability, especially:

- exact `premises_used` quality for Type 1
- generated-code reliability for both tracks
- latency and cold-start behavior on the deployed endpoint

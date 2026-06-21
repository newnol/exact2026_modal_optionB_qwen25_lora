# EXACT 2026 — Solution Description

## Datasets Used

**Type 1 – Logic.** Official EXACT 2026 Logic Dataset. State the number of raw, augmented, train, validation, and test samples used. Include 1-2 compact sample entries.

**Type 2 – Physics.** Official EXACT 2026 Physics Dataset. State the number of valid records, SFT samples, and GRPO/RL samples used. Include 1-2 compact sample entries.

## Approach and Method

We implement a hybrid neuro-symbolic pipeline with a zero-latency router.

**Stage 1 — Routing.** The `/predict` endpoint reads the `type` field. Type 1 requests are routed to the logic LoRA. Type 2 requests are routed to deterministic physics solvers first and then to the physics LoRA if needed.

**Stage 2 — Reasoning.**

- **Logic:** `Qwen/Qwen2.5-7B-Instruct + type1-logic LoRA` generates the answer, used premise indices, and structured reasoning.
- **Physics:** deterministic solvers handle common exact patterns. Otherwise `Qwen/Qwen2.5-7B-Instruct + type2-physics LoRA` generates a standalone Python/SymPy solver, which is executed in a short-timeout sandbox. The sandbox result overwrites the model prediction on success. If sandbox execution fails, the system falls back to direct JSON answer generation from the physics LoRA.

**Stage 3 — Explanation Generation.** Type 1 returns a concise explanation with premise indices. Type 2 returns a verified explanation from the sandbox result when available.

Pipeline summary:

```txt
INPUT -> /predict router
  -> Type1: Qwen2.5-7B + type1-logic LoRA -> JSON output
  -> Type2: deterministic solver -> Qwen2.5-7B + type2-physics LoRA -> sandbox/JSON output
```

Both LoRA adapters are loaded before the grading slot. The system does not load, unload, or swap LLMs during a query. Routing only selects the already-loaded adapter by `model` id.

## Model Size Calculation

- Base LLM: `Qwen/Qwen2.5-7B-Instruct` = 7.61B parameters.
- LoRA 1: `NguyenAn05/qwen2.5-type1-grpo-lora`, rank r=64 = 0.167B trainable/adapter parameters.
- LoRA 2: `not-a-real-ai-guy/qwen2.5-type2-option-b-modes-lora`, rank r=64 = 0.167B trainable/adapter parameters.

Both LoRAs share one base model instance and are loaded together in one vLLM server.

```txt
Active per request: 7.61B base + 0.167B selected LoRA = 7.78B
Total loaded:       7.61B base + 0.167B LoRA1 + 0.167B LoRA2 = 7.95B
```

The total loaded LLM capacity remains within the 8B-class limit. Non-LLM tools such as deterministic solvers and sandbox execution do not count toward the LLM parameter limit.

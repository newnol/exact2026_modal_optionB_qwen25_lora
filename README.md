# EXACT 2026 Modal Option B — Qwen2.5 Base + Two LoRAs

This project deploys an EXACT 2026-compliant API with:

- `POST /predict` — one wrapper endpoint for both Type 1 and Type 2.
- `GET /v1/models` — vLLM model verification endpoint.
- One shared base model: `Qwen/Qwen2.5-7B-Instruct`.
- Two LoRA adapters loaded at vLLM startup:
  - `type1-logic` = `NguyenAn05/qwen2.5-type1-grpo-lora`
  - `type2-physics` = `not-a-real-ai-guy/qwen3-type2-option-b-modes-grpo-lora`

There is no model swapping during a query. `/predict` only selects the already-loaded LoRA by setting the OpenAI `model` field to `type1-logic` or `type2-physics`.

## Important preflight

Both LoRAs must be trained from the same base model:

```bash
python scripts/preflight_check_loras.py
```

Expected result:

```txt
PASS: both LoRA adapters match the shared base. You can deploy Option B.
```

If the Type 2 adapter still declares a Qwen3 base in `adapter_config.json`, rebuild/retrain it on `Qwen/Qwen2.5-7B-Instruct` before deploying this Option B project.

## Deploy (all-in-one)

```bash
bash scripts/deploy.sh
```

This runs preflight → deploy → prints instructions to get URLs.

## Logging system

Every request is logged to a Modal Volume (`exact2026-optionb-logs`) as JSONL with three event types:

| Event | When | What's logged |
|---|---|---|
| `predict` | Start & end of `/predict` | request params, response, latency, **GPU/RAM/CPU metrics before & after** |
| `llm_call` | Each vLLM chat completion | system_prompt, user_prompt, raw_response, latency |
| `premise_audit` | When premise auditor corrects the model | model_gave vs auditor_fixed (type1 only) |

### Download and analyze

```bash
# 1. Download logs
uv run modal volume get exact2026-optionb-logs requests.jsonl > logs.jsonl

# 2. Analyze
python scripts/analyze_logs.py logs.jsonl
```

Example output:

```
Total entries: 24
Events:
  predict: 8
  llm_call: 8
  premise_audit: 2

Predict (8):
  Latency: min=1.23s  max=8.56s  avg=3.42s
  HW (first request):
    GPU: L40S util=45% mem=8192/23034 MiB temp=62°C
    RAM: 62.3% used

LLM calls (8):
  Latency: min=0.95s  max=7.80s  avg=3.10s

Premise auditor corrections (2):
  type1_yesno: model=[1,3,4,5,7] → auditor=[0,1,2,3,4,5]
  type1_text:  model=[0,1,4,5]   → auditor=[0,1,3,4,5]
```

### Full integration test

```bash
bash scripts/test_full.sh
```

Sends 6 type1 questions (yes/no, uncertain, number, text, multiple choice, custom logic) and 2 type2 physics problems. Validates each response is valid JSON.

### Log file format

Each line is a JSON object with a `"event"` field. Logs are written to `/root/exact/logs/requests.jsonl` inside the Modal container, which is backed by the Modal Volume `exact2026-optionb-logs`. If the log path is unwritable (e.g. local tests), logging is silently skipped.

## Local API shape test without GPU

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MOCK_MODE=true pytest -q
```

## Modal deployment

```bash
pip install -r requirements-modal-local.txt
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxx
python scripts/preflight_check_loras.py
modal deploy modal_exact2026.py
modal run modal_exact2026.py
```

The last command prints:

```txt
Prediction URL: https://...modal.run/predict
vLLM models URL: https://...modal.run/v1/models
```

Put those two URLs into `submission/urls.txt`.

## Public endpoint test

```bash
export PREDICT_URL="https://<workspace>--exact2026-optionb-qwen25-predict-api.modal.run/predict"
export VLLM_MODELS_URL="https://<workspace>--exact2026-optionb-qwen25-vllm-server.modal.run/v1/models"

bash scripts/modal_test_public.sh
python scripts/modal_latency_check.py
```

`/v1/models` must include:

```txt
Qwen/Qwen2.5-7B-Instruct
type1-logic
type2-physics
```

## Keep warm during grading

Run 10-20 minutes before your grading slot:

```bash
python scripts/modal_keep_warm_on.py
```

After grading, stop warm containers:

```bash
python scripts/modal_keep_warm_off.py
```

## Build submission package

```bash
cp /path/to/solution.pdf submission/solution.pdf
# edit submission/urls.txt with real Modal URLs
bash scripts/build_submission_package.sh your_team_name
```

Output:

```txt
dist_submission/your_team_name.zip
```

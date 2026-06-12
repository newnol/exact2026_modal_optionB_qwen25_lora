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

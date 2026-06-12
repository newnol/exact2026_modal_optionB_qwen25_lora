# Modal Deploy Guide — EXACT 2026 Option B

## 1. Install and login

```bash
pip install -r requirements-modal-local.txt
modal setup
```

## 2. Create Hugging Face secret

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

If the secret already exists, edit it in the Modal dashboard.

## 3. Verify LoRA bases

```bash
python scripts/preflight_check_loras.py
```

Do not deploy if this fails. Option B requires both Type 1 and Type 2 LoRA adapters to match `Qwen/Qwen2.5-7B-Instruct`.

## 4. Deploy

```bash
modal deploy modal_exact2026.py
modal run modal_exact2026.py
```

Save the printed URLs into `submission/urls.txt`:

```txt
Prediction endpoint:
https://...modal.run/predict

vLLM model endpoint:
https://...modal.run/v1/models
```

## 5. Test

```bash
export PREDICT_URL="https://...modal.run/predict"
export VLLM_MODELS_URL="https://...modal.run/v1/models"

bash scripts/modal_test_public.sh
python scripts/modal_latency_check.py
```

## 6. Grading slot operations

Before slot:

```bash
python scripts/modal_keep_warm_on.py
```

After slot:

```bash
python scripts/modal_keep_warm_off.py
```

## If vLLM version pin fails

Open `modal_exact2026.py` and replace:

```python
"vllm==0.21.0"
```

with:

```python
"vllm"
```

Then redeploy.

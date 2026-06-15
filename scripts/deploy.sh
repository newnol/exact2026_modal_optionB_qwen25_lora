#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Create log volume (Modal will auto-create if missing) ==="
uv run modal volume create exact2026-optionb-logs 2>/dev/null || true

echo ""
echo "=== 2. Preflight: Checking LoRA adapter bases ==="
uv run python scripts/preflight_check_loras.py

echo ""
echo "=== 3. Deploying to Modal ==="
uv run modal deploy modal_exact2026.py

echo ""
echo "=== 4. Get URLs ==="
echo "Run this to print endpoint URLs:"
echo ""
echo "  uv run modal run modal_exact2026.py"
echo ""
echo "Save them to .env as:"
echo "  PREDICT_URL=https://...modal.run/predict"
echo "  VLLM_MODELS_URL=https://...modal.run/v1/models"
echo ""
echo "Then verify:"
echo "  bash scripts/modal_test_public.sh"
echo ""
echo "To download logs:"
echo "  uv run modal volume ls exact2026-optionb-logs"
echo "  uv run modal volume get exact2026-optionb-logs requests.jsonl"

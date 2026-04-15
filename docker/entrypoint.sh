#!/usr/bin/env bash
# ------------------------------------------------------------------
# AES Thesis trainer entrypoint
# Prints a short environment banner then execs the user command.
# Captures GPU topology and CUDA/torch versions for run logs so that
# the Reproducibility reviewer (talep 13c) can cross-check against
# ADR Annex B image digest.
# ------------------------------------------------------------------
set -euo pipefail

echo "============================================================"
echo "  AES Thesis — Trainer Container"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# Quick sanity: GPU + CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader || true
else
    echo "[WARN] nvidia-smi not present — container may not see GPU"
fi

python - <<'PY'
import os, platform
try:
    import torch
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device : {torch.cuda.get_device_name(0)}")
        print(f"cuda arch   : {torch.cuda.get_device_capability(0)}")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")
    try:
        import flash_attn
        print(f"flash_attn  : {flash_attn.__version__}")
    except Exception as e:
        print(f"flash_attn  : NOT AVAILABLE ({e})")
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except Exception:
        pass
except ImportError:
    print("torch not installed (image build issue?)")
print(f"python      : {platform.python_version()}")
print(f"platform    : {platform.platform()}")
PY

echo "------------------------------------------------------------"
exec "$@"

# ------------------------------------------------------------------
# AES Thesis — Training image for ASUS Ascent GX10 (GB10, ARM64, sm_100)
# Base: NVIDIA NGC PyTorch container (ARM64-native, prebuilt Flash-Attn-2,
#       CUDA 12.6, cuDNN, BF16 kernels, torch.compile ready).
#
# Rationale (ADR-001-rev2 §5 / talep 13b-1 & 13c-1):
#   * NGC wheels compiled for Grace Blackwell → avoids ARM64 flash-attn
#     compile hell.
#   * Image digest (SHA256) pinned in ADR Annex B → reproducibility by
#     container, not by requirements-lock alone.
#   * Same image runs on GX10 dev box and any NGC-compatible cloud GPU.
# ------------------------------------------------------------------
ARG NGC_TAG=25.12-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

LABEL org.opencontainers.image.title="aes-thesis-trainer"
LABEL org.opencontainers.image.source="https://github.com/<your-org>/aes-thesis"
LABEL org.opencontainers.image.description="AES training image (GX10 target)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    MLFLOW_TRACKING_URI=file:/workspace/mlruns

# System packages — minimal, NGC image already ships gcc/git/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- Python deps -------------------------------------------------
# NGC image already has: torch, transformers (old), flash-attn, triton.
# We layer only the project-specific deps on top.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-deps -r /tmp/requirements.txt

# ---- Project source ---------------------------------------------
# Mount-time override is recommended (docker-compose volume) so that
# edits in dev sync live. The COPY here is the fallback for
# production-style self-contained images.
COPY . /workspace

# ---- Entrypoint --------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

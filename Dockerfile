# Mirror of bon5co/mineru-railway, which is what the published template deploys
# (ghcr.io/bon5co/mineru-railway:latest). Kept here so the repo convention holds:
# railway.toml can reference neither a prebuilt image nor a volume, so the
# wrapper Dockerfile is the only way to pin the upstream stack in-repo.
#
# There is no official CPU image for mineru-api -- upstream's Dockerfile is built
# on vllm/vllm-openai and assumes a GPU -- so this builds the CPU stack directly
# rather than wrapping an upstream tag.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        fonts-noto-core \
        fonts-noto-cjk \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tini \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# torch and torchvision come from PyTorch's CPU index BEFORE mineru is installed.
# `pip install mineru[core]` on its own resolves torch from PyPI, which is the
# CUDA build: it drags in the whole nvidia-* wheel set (cuBLAS, cuDNN, NCCL,
# cuSPARSE, ...) that a Railway container has no device for. Installing the CPU
# wheels first leaves mineru's `torch>=2.6.0,<3` already satisfied, so the second
# pip call does not reach for the CUDA one.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.6.0,<3" torchvision

# Pinned, not `>=3.0.0`. MinerU changed its default backend and its model set
# between minor versions; an unpinned template rebuilds into a different parser
# every time someone deploys it.
RUN pip install "mineru[core]==3.4.4" \
    && python -c "import torch; assert not torch.version.cuda, torch.version.cuda; print('torch', torch.__version__, 'cpu-only OK')"

# --- Bake the pipeline models into the image --------------------------------
#
# Upstream (and the competing template) download ~1.5 GB of layout/OCR/table/
# formula models at runtime: either on first boot or, worse, lazily on the first
# request, which stalls it for 60-120s behind Railway's 5-minute edge timeout and
# makes the deploy's success depend on Hugging Face being reachable.
#
# Downloading at build time and then pinning MINERU_MODEL_SOURCE=local means the
# container needs no network at all to parse, the first request is as fast as the
# thousandth, and there is no first-boot window for a healthcheck to kill.
ENV MINERU_MODELS_HOME=/opt/mineru \
    MINERU_TOOLS_CONFIG_JSON=/opt/mineru/mineru.json \
    HF_HOME=/opt/mineru/hf
RUN mkdir -p /opt/mineru \
    && MINERU_MODEL_SOURCE=huggingface mineru-models-download -s huggingface -m pipeline \
    && python - <<'PY'
import json, os
cfg = json.load(open(os.environ["MINERU_TOOLS_CONFIG_JSON"]))
assert cfg.get("models-dir", {}).get("pipeline"), cfg
print("baked pipeline models at", cfg["models-dir"]["pipeline"])
PY

# Resolve models from the baked directory, never from the network.
ENV MINERU_MODEL_SOURCE=local

COPY railway_mineru_api.py /usr/local/bin/railway_mineru_api.py
COPY railway-entrypoint.sh /usr/local/bin/railway-entrypoint.sh
RUN chmod +x /usr/local/bin/railway-entrypoint.sh

RUN mkdir -p /data
ENV MINERU_DATA_DIR=/data

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/railway-entrypoint.sh"]

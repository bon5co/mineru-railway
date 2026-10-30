# mineru-railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/mineru-v344-or-pdf-to-markdown-api-for-r?referralCode=Z1xivh&utm_medium=integration&utm_source=template&utm_campaign=generic)

**Deploy on Railway:** [MinerU template](https://railway.com/deploy/mineru-v344-or-pdf-to-markdown-api-for-r?referralCode=Z1xivh&utm_medium=integration&utm_source=template&utm_campaign=generic)

Wrapper image behind the Railway template **MinerU — Document Parsing API (CPU)**.

Published as `ghcr.io/bon5co/mineru-railway:latest`.

[MinerU](https://github.com/opendatalab/MinerU) is OpenDataLab's document parsing
engine: PDFs, images and Office documents in, LLM-ready Markdown and JSON out,
with layout detection, OCR in 80+ languages, table structure and formula
recognition. This image runs its `mineru-api` FastAPI server in CPU pipeline mode
on Railway.

Upstream ships no CPU image — its Dockerfile is built on `vllm/vllm-openai` and
assumes a GPU — so this builds the CPU stack directly instead of wrapping a tag.

## What this image changes, and why

### 1. `pipeline` is the server's default backend

MinerU 3.4's `DEFAULT_BACKEND` is `hybrid-engine`, which needs a local VLM and a
GPU. On a CPU container every request that omits `backend=` fails, including the
"Try it out" button in the service's own Swagger UI. It is a module constant in
`mineru/cli/backend_options.py` read through a `from ... import`, so no
environment variable can change it — `railway_mineru_api.py` rebinds it before
`mineru.cli.fast_api` is imported.

The result is that a plain `POST /file_parse` with just a file works.

### 2. Threads are sized from the cgroup, not the host

MinerU's pipeline backend is torch end to end. torch sizes its intra-op thread
pool from the cores it can *see*, which inside a container is the host's core
count rather than the cgroup quota, so on Railway it oversubscribes badly.
`railway-entrypoint.sh` reads `/sys/fs/cgroup/cpu.max` and exports
`OMP_NUM_THREADS` (plus the MKL/OpenBLAS/NumExpr equivalents) before Python
imports torch, which is the only point at which torch reads it.

### 3. Models are baked into the image

Upstream downloads ~1.5 GB of layout, OCR, table and formula models at runtime —
either on first boot or lazily on the first request, which stalls that request
for 60–120s and makes the deploy depend on Hugging Face being reachable. Here
they are downloaded at build time and `MINERU_MODEL_SOURCE=local` is pinned, so
the container needs no network to parse and the first request is as fast as the
thousandth.

### 4. CPU-only torch

`pip install mineru[core]` resolves torch from PyPI, which is the CUDA build: it
pulls in ~2.7 GB of `nvidia-*` wheels (cuBLAS, cuDNN, NCCL, cuSPARSE, …) that a
CPU container can never load. Installing the CPU wheels from PyTorch's own index
first leaves `torch>=2.6.0,<3` already satisfied, and the build asserts
`torch.version.cuda is None` so a resolver change cannot silently undo it.

### 5. An API key gate

`mineru-api` installs exactly one middleware upstream, GZip. There is no
authentication of any kind, and a Railway template hands the deployer a public
HTTPS URL — an open document-parsing endpoint is a free-compute faucet. When
`MINERU_API_KEY` is set, `railway_mineru_api.py` requires it as
`Authorization: Bearer <key>` or `X-API-Key: <key>` on every route except
`/health` and the docs. Unset, the gate is skipped and a warning is logged.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MINERU_API_KEY` | *(unset)* | Required on every request when set. The template generates it with `${{secret(32)}}`. |
| `PORT` | `8000` | Injected by Railway; the server binds it. |
| `MINERU_DATA_DIR` | `/data` | Volume mount; async task output lives under `$MINERU_DATA_DIR/output`. |
| `OMP_NUM_THREADS` | cgroup CPU quota | Set explicitly to override the entrypoint's calculation. |
| `MINERU_DEVICE_MODE` | `cpu` | Pinned; there is no GPU on a Railway container. |
| `MINERU_MODEL_SOURCE` | `local` | Baked models. Set to `huggingface` to fetch instead. |

## Build

CI builds and pushes on every push to `main` (`.github/workflows/build.yml`).

FROM rocm/dev-ubuntu-24.04:7.2.4-complete
ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install Python and other dependencies
RUN apt-get update && apt upgrade -y && apt-get install -y --no-install-recommends \
    espeak-ng \
    espeak-ng-data \
    rocrand \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    wget \
    nano \
    g++ \
    zstd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \

    # Install UV using the installer script
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/ \

    # Create non-root user and set up directories and permissions
    && useradd -m -u 1001 appuser \
    && mkdir -p /app/api/src/models/v1_0 \
    && chown -R appuser:appuser /app \
    # Models folder
    && mkdir -p /app/api/src/models/v1_0

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install dependencies with GPU extras (using cache mounts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 && \
    uv sync --extra rocm

# Japanese support requires the UniDic dictionary (~526MB on disk) for fugashi/MeCab.
# Enabled by default; set --build-arg INCLUDE_JAPANESE=false to skip and shave the image.
# Keep true. Setting false silently breaks ja generation.
ARG INCLUDE_JAPANESE=true
RUN if [ "$INCLUDE_JAPANESE" = "true" ]; then \
        .venv/bin/python -m unidic download; \
    fi

# gfx1151 (Strix Halo) requires ROCm 7.2+ and AMD's own PyTorch wheels.
# The pytorch.org ROCm 7.2 wheels are built against ROCm 6.x libraries and segfault.
# Reinstall torch and triton from AMD's repo.radeon.com wheel index.
# This is done AFTER uv sync so the lockfile stays intact as the source of truth
# for all other packages. See: https://github.com/remsky/Kokoro-FastAPI/issues/454
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --force-reinstall --no-deps \
    torch==2.8.0+rocm7.2.4.lw.git6bea3e0b \
    triton==3.3.1+rocm7.2.4.git28a7371e \
    --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/

# Skip kdb_install.sh and rocblas replacement:
# - gfx1151 has no precompiled MIOpen KDB files in ROCm apt repositories.
# - MIOpen JIT-compiles kernels and caches them locally (~/.config/miopen, ~/.cache/miopen).
# - The rocblas replacement step targets Arch Linux rocblas-6.4.4-1 which is incompatible
#   with ROCm 7.2.4 and lacks gfx1151 kernels.
# - Use docker/rocm/warmup_miopen.py to pre-populate the MIOpen cache after first run.

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser docker/scripts/ ./
COPY --chown=appuser:appuser docker/rocm/warmup_miopen.py /app/docker/rocm/warmup_miopen.py

RUN chmod +x ./entrypoint.sh

# Pre-create MIOpen cache dirs so named-volume mounts inherit appuser
# ownership (Docker would otherwise create the mount target as root:root
# and MIOpen as uid 1001 could not write to it on first run).
RUN mkdir -p /home/appuser/.config/miopen /home/appuser/.cache/miopen

# Set all environment variables in one go.
# HSA_OVERRIDE_GFX_VERSION=11.5.1: gfx1151 is not in ROCm's official device list.
# Without this override the HSA runtime fails to initialize or falls back to CPU.
# HIP_VISIBLE_DEVICES=0: pins HIP to the first (and typically only) GPU on Strix Halo.
# MIOPEN_FIND_MODE=2 (FAST): reuse the on-disk find DB when present, fall back
# to immediate-mode heuristics otherwise. Cheap default that avoids the 5-60s
# per-shape kernel search on first hit. See docker/rocm/warmup_miopen.py to
# pre-populate the cache for best-case latency. Override at runtime to redo
# the search (e.g. MIOPEN_FIND_MODE=3 + MIOPEN_FIND_ENFORCE=3).
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="gpu" \
    MIOPEN_FIND_MODE=2 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    HIP_VISIBLE_DEVICES=0 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]

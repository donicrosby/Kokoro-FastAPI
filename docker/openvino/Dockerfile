FROM openvino/ubuntu24_dev:2025.3.0

USER root
# Install dependencies and check espeak location
# Rust is required to build sudachipy and pyopenjtalk-plus
RUN apt-get update -y &&  \
    apt-get install -y espeak-ng espeak-ng-data git libsndfile1 curl ffmpeg g++ cmake python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data && \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    usermod -l appuser openvino && \
    groupmod -n appuser openvino && \
    usermod -d /home/appuser -m appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Install Rust for the non-root user so builds (e.g., sudachipy) succeed
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Ensure Cargo and the Python venv are on PATH; extend HTTP timeouts for uv
ENV PATH="/home/appuser/.cargo/bin:/app/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=3 \
    VIRTUAL_ENV=

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with OpenVINO extras
RUN uv venv --python 3.12 && \
    uv sync --extra openvino --no-cache

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh
RUN source "${INTEL_OPENVINO_DIR}"/setupvars.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="openvino"

ENV DOWNLOAD_MODEL=true
# Download model if enabled
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi


# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]

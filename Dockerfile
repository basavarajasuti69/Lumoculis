# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

# Install uv for reproducible dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency metadata first for layer caching.
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev tools).
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-cache

# Runtime stage.
FROM python:${PYTHON_VERSION}-slim AS runtime

# Runtime dependencies for OpenCV (libGL, libSM, libXrender, libXext).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libfontconfig1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the application.
RUN groupadd -r lumoculis && useradd -r -g lumoculis -d /app -s /sbin/nologin lumoculis

WORKDIR /app

# Copy the installed packages from the builder stage.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source and configuration.
COPY src/lumoculis ./src/lumoculis
COPY pyproject.toml ./
COPY app.py ./

# Make /tmp writable for temporary Grad-CAM output.
RUN chown -R lumoculis:lumoculis /app /tmp

USER lumoculis

# Streamlit runs on this port.
ENV PORT=8501
ENV LUMOCULIS_DEVICE=cpu
ENV LUMOCULIS_MAX_UPLOAD_MB=10

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/', timeout=5)" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]

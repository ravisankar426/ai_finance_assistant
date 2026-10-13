# One image for the API, the Streamlit UI, and the index builder (REQ-NFR-06).
# Secrets are never baked in: keys come from the environment at run time (.env via compose).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    APP__LOG_FORMAT=json

WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY config.yaml ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen \
    && useradd --create-home --uid 1000 app \
    && mkdir -p data src/data/index src/data/embedding_cache \
    && chown -R app:app data src/data/index src/data/embedding_cache

USER app
EXPOSE 8000 8501

# Default: the API. docker-compose overrides the command for the UI and the index builder.
CMD ["uvicorn", "src.web_app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

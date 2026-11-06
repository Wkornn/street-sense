FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies before copying the code, so this slow layer stays cached
# when only the code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

# 0.0.0.0 so the app is reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# ---- Stage 1: Build frontend ----
FROM node:22-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ---- Stage 2: Production image ----
FROM python:3.11-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copy Python dependency files first (cache layer)
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Install build tools for native extensions (psutil, etc.) then Python dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc python3-dev \
  && rm -rf /var/lib/apt/lists/* \
  && cd backend && uv sync --frozen --no-dev

# Copy backend source
COPY backend/ ./backend/

# Copy built frontend into backend static serving path
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 5001

CMD ["backend/.venv/bin/python", "backend/run.py"]

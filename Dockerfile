# Multi-stage build for minimal image size
# Stage 1: Builder - Compile dependencies
FROM python:3.13-slim AS builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code for installation
COPY pyproject.toml README.md LICENSE ./
COPY rice_price_prediction ./rice_price_prediction

# Install Python dependencies (use regular install, not editable)
RUN pip install --no-cache-dir .

# Stage 2: Runtime - Final lightweight image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies (libgomp for OpenMP support)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder (without build tools)
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY rice_price_prediction ./rice_price_prediction

# Create directories for models and data (mounted as volumes at runtime)
RUN mkdir -p /app/models /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run the application
CMD ["uvicorn", "rice_price_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

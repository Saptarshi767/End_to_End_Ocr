# Multi-stage Docker build for OCR Table Analytics

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Stage 2: Python dependencies
FROM base as dependencies

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Application
FROM dependencies as application

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY migrations/ /app/migrations/
COPY alembic.ini /app/
COPY deployment/docker-entrypoint.sh /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/uploads /app/exports && \
    chown -R appuser:appuser /app

# Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# Switch to app user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["api"]
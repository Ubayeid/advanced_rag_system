# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Removed --no-cache-dir to enable pip caching for faster rebuilds
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Ensure templates directory is present
RUN mkdir -p templates && cp -r templates/* templates/ || true

# Create necessary directories
RUN mkdir -p storage data cache

# Set permissions
RUN chmod +x main.py

# Expose port for the web interface
EXPOSE 8000

# Create a non-root user for security and switch to it
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Set default environment variables
ENV EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
ENV EMBEDDING_DIM=384
ENV LLM_MODEL_NAME="gpt-3.5-turbo"
ENV LLM_TEMPERATURE=0.7
ENV LLM_MAX_TOKENS_CONTEXT=4000
ENV LLM_MAX_TOKENS_RESPONSE=1000
ENV TARGET_CHUNK_SIZE_TOKENS=512
ENV CHUNK_OVERLAP_SENTENCES=2
ENV MIN_CHUNK_SIZE_TOKENS=20
ENV STORAGE_PATH="./storage"
ENV DATA_PATH="./data"
ENV ENABLE_CACHE="true"
ENV CACHE_DIR="./cache"
ENV MAX_CONCURRENT_REQUESTS=10
ENV REQUEST_TIMEOUT=60
ENV CACHE_VALIDITY_DAYS=1
ENV SPACY_MODEL="en_core_web_sm"

# Health check (uncomment if you want to enable this)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command to run the Flask application using Gunicorn
# This replaces `python web_server.py` with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "web_server:app"]
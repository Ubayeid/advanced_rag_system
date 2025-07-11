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
    # Add any other system libraries Neo4j driver might implicitly need, e.g., libffi-dev if hitting CFFI errors
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Ensure templates directory is present (if it exists, copy its contents)
RUN mkdir -p templates && cp -r templates/* templates/ || true

# Create necessary directories
RUN mkdir -p storage data cache

# Set permissions
# Note: chmod +x main.py might not be needed if running via gunicorn
# Ensure app user has write permissions for mounted volumes
RUN chown -R app:app /app/storage /app/data /app/cache || true # Grant ownership to app user, non-fatal if mounts don't exist yet

# Expose port for the web interface
EXPOSE 8000

# Create a non-root user for security and switch to it
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Default command to run the Flask application using Gunicorn
# This will be overridden by docker-compose.yml's command, but good for direct docker run
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "web_server:app"]
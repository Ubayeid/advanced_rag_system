# Docker Setup for CIROH RAG System

This document explains how to run the CIROH Hybrid RAG System using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed
- OpenAI API key (for LLM functionality)

## Quick Start

### 1. Clone and Navigate to Project

```bash
cd /path/to/ciroh_new
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your OpenAI API key
# OPENAI_API_KEY=your_actual_openai_api_key_here
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key (required for LLM generation)

**Optional Environment Variables (with defaults):**
- `EMBEDDING_MODEL_NAME`: all-MiniLM-L6-v2
- `EMBEDDING_DIM`: 384
- `LLM_MODEL_NAME`: gpt-3.5-turbo
- `LLM_TEMPERATURE`: 0.7
- `LLM_MAX_TOKENS_CONTEXT`: 4000
- `LLM_MAX_TOKENS_RESPONSE`: 1000
- `TARGET_CHUNK_SIZE_TOKENS`: 512
- `CHUNK_OVERLAP_SENTENCES`: 2
- `MIN_CHUNK_SIZE_TOKENS`: 20
- `STORAGE_PATH`: ./storage
- `DATA_PATH`: ./data
- `ENABLE_CACHE`: true
- `CACHE_DIR`: ./cache
- `MAX_CONCURRENT_REQUESTS`: 10
- `REQUEST_TIMEOUT`: 60
- `CACHE_VALIDITY_DAYS`: 1
- `SPACY_MODEL`: en_core_web_sm

### 3. Prepare Your Data

Place your documents (`.txt` or `.pdf` files) in the `./data` directory:

```bash
mkdir -p data
# Copy your documents to the data directory
cp /path/to/your/documents/*.pdf data/
cp /path/to/your/documents/*.txt data/
```

### 4. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Run the application
docker-compose up
```

Or run in detached mode:
```bash
docker-compose up -d
```

### 5. Access the Application

The application will start and provide an interactive command-line interface. You can interact with it by:

```bash
# If running in detached mode, attach to the container
docker-compose exec ciroh-rag python main.py

# Or run interactively
docker-compose run --rm ciroh-rag python main.py
```

## Alternative: Direct Docker Commands

### Build the Image

```bash
docker build -t ciroh-rag .
```

### Run the Container

```bash
docker run -it \
  --name ciroh-rag-system \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/cache:/app/cache \
  -e OPENAI_API_KEY=your_api_key_here \
  ciroh-rag
```

## Volume Mounts

The Docker setup uses the following volume mounts for data persistence:

- `./data:/app/data`: Input documents directory
- `./storage:/app/storage`: Processed knowledge base storage
- `./cache:/app/cache`: Cache directory for embeddings and models

## Managing the Application

### View Logs

```bash
docker-compose logs -f ciroh-rag
```

### Stop the Application

```bash
docker-compose down
```

### Restart the Application

```bash
docker-compose restart
```

### Remove Everything (including volumes)

```bash
docker-compose down -v
docker system prune -f
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set" error:**
   - Ensure your `.env` file contains a valid OpenAI API key
   - Check that the `.env` file is in the project root directory

2. **Permission errors with volume mounts:**
   - Ensure the `data`, `storage`, and `cache` directories exist and have proper permissions
   - On Windows, ensure Docker Desktop has access to the project directory

3. **Out of memory errors:**
   - The application can be memory-intensive. Ensure Docker has sufficient memory allocated
   - Consider reducing `MAX_CONCURRENT_REQUESTS` in your `.env` file

4. **Slow startup:**
   - The first run will download models and process documents, which can take time
   - Subsequent runs will be faster due to caching

### Health Checks

The container includes health checks. You can monitor the health status:

```bash
docker-compose ps
```

### Resource Usage

Monitor resource usage:

```bash
docker stats ciroh-rag-system
```

## Development

### Rebuilding After Code Changes

```bash
docker-compose build --no-cache
docker-compose up
```

### Running Tests

```bash
docker-compose run --rm ciroh-rag python -m pytest
```

## Production Considerations

For production deployment:

1. **Security:**
   - Use Docker secrets for sensitive environment variables
   - Run the container as a non-root user (already configured)
   - Consider using a reverse proxy for external access

2. **Performance:**
   - Allocate sufficient memory and CPU resources
   - Use persistent volumes for data storage
   - Consider using GPU-enabled images for faster embedding generation

3. **Monitoring:**
   - Set up logging aggregation
   - Monitor resource usage
   - Implement proper health checks

## Support

For issues related to the Docker setup, check:
- Docker logs: `docker-compose logs ciroh-rag`
- Container status: `docker-compose ps`
- Resource usage: `docker stats ciroh-rag-system`

## Notes and Tips

- **Service Name:**
  - The service name used in commands like `docker-compose exec ciroh-rag` or `docker-compose logs -f ciroh-rag` should match the name defined in your `docker-compose.yml` file. If your service is named differently (e.g., `app`), replace `ciroh-rag` with your actual service name.

- **Windows Users:**
  - When running Docker commands with volume mounts, use `%cd%` instead of `$(pwd)` in PowerShell or Command Prompt:
    ```powershell
    docker run -it ^
      --name ciroh-rag-system ^
      -v %cd%\data:/app/data ^
      -v %cd%\storage:/app/storage ^
      -v %cd%\cache:/app/cache ^
      -e OPENAI_API_KEY=your_api_key_here ^
      ciroh-rag
    ```

- **Reprocessing Data:**
  - If you add, remove, or modify files in the `data` directory, simply stop and restart the container (`docker-compose down` then `docker-compose up`). The system will automatically detect changes and reprocess the knowledge base.

- **Interactive Mode:**
  - If you want to interact with the CLI after starting in detached mode, use:
    ```bash
    docker-compose exec <service-name> python main.py
    ```
    Replace `<service-name>` with your actual service name from `docker-compose.yml`.
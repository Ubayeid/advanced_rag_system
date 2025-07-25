# CIROH Knowledge Graph & RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system for hydrological and scientific document QA, featuring hybrid vector and knowledge graph retrieval, gap analysis, and interactive visualization.

---

## Overview

This project integrates advanced NLP, knowledge graph construction, and retrieval-augmented generation to unify, analyze, and query hydrology-related knowledge. It supports:
- **Automated data ingestion and entity extraction**
- **Knowledge graph construction and visualization**
- **Hybrid retrieval (vector + graph) for robust question answering**
- **Systematic knowledge gap analysis to identify research opportunities**

---

## Features

- **Hybrid Retrieval:** Combines vector search (FAISS) and knowledge graph traversal for robust, context-aware retrieval.
- **Knowledge Graph:** Entity and relation extraction, gap analysis, and interactive visualization (Plotly, Matplotlib).
- **Document Processing:** Supports PDF and TXT ingestion, chunking, and embedding.
- **LLM Integration:** Uses OpenAI GPT models for answer generation with strict source citation.
- **REST API:** Flask-based API with endpoints for querying and system stats.
- **Dockerized:** Easy deployment with Docker and Docker Compose.
- **Configurable:** Environment variables for model selection, chunking, and more.

---

## Workflow

1. **Data Preparation**
   - Place your PDF or TXT documents in the `data/` directory.
   - The system cleans, preprocesses, and extracts entities/relationships using NLP.

2. **Knowledge Graph Construction**
   - Entities and relationships are mapped into a Neo4j-based knowledge graph.
   - Cross-document integration merges duplicate entities for a unified view.

3. **Query System**
   - Users can query the system via REST API or web UI.
   - The system uses both vector similarity and graph traversal to find relevant answers.

4. **Response Generation**
   - Answers are generated using LLMs, with supporting evidence and source citations.
   - Results are contextualized and visualized as needed.

5. **Knowledge Gap Analysis**
   - The system analyzes the graph for sparsely connected or critical nodes.
   - Generates reports and visualizations in the `storage/` directory, highlighting research gaps and integration needs.

---

## Directory Structure

```
ciroh_new/
├── main.py                # Core RAG components and logic
├── web_server.py          # Flask API server and RAG system initialization
├── knowledge_graph.py     # Knowledge graph visualization and gap analysis
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build instructions
├── docker-compose.yml     # Multi-container orchestration
├── data/                  # Input documents (PDF/TXT)
├── storage/               # Persisted vector index, KG, and reports
├── cache/                 # Optional cache for embeddings, etc.
└── README.md              # This file
```

---

## Quick Start

### 1. Prepare Data

- Place your PDF or TXT documents in the `data/` directory.

### 2. Set Environment Variables

Create a `.env` file in the project root (or set variables in your shell):

```env
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=gpt-3.5-turbo
# ...other variables as needed (see docker-compose.yml)...
```

### 3. Build and Run with Docker Compose

```sh
docker-compose up --build
```

- The API will be available at [http://localhost:8000](http://localhost:8000)

### 4. Access the Web UI

- Open [http://localhost:8000](http://localhost:8000) in your browser for the chat interface.

---

## API Endpoints

- `POST /query`  
  **Body:** `{ "query": "your question here" }`  
  **Response:** JSON with answer, confidence, sources, and query analysis.

- `GET /stats`  
  Returns system statistics (index size, KG nodes/edges, etc.).

---

## Knowledge Graph Visualization & Gap Analysis

- After document processing, an interactive knowledge graph HTML and a gap analysis report are generated in the `storage/` directory.
- Visualization generation can be toggled with the `GENERATE_KG_VISUALIZATION` environment variable.
- The gap analysis report highlights:
  - Sparsely connected or isolated entities (potential research gaps)
  - Critical nodes and relationships
  - Cross-document integration needs
  - Actionable recommendations and metrics

---

## Troubleshooting

- If you see **"RAG system not initialized"** or **"RAG system is not ready"**:
  - Check the container logs for errors (`docker logs ciroh-rag-system`).
  - Ensure all environment variables and required models are available.
  - Make sure your `data/` directory contains at least one valid document.

- To debug initialization:
  - Enter the container:  
    `docker exec -it ciroh-rag-system /bin/bash`
  - Run:  
    `python main.py`  
    (Add debug code to `main.py` if needed.)

---

## Development Notes

- All core logic is in `main.py` (retrieval, embedding, KG, etc.).
- The Flask API and RAG system initialization are in `web_server.py`.
- Knowledge graph analysis and visualization are in `knowledge_graph.py`.
- The system expects documents in `data/` and persists processed data in `storage/`.

---

## License

MIT License (add your license here)

---

## Acknowledgments

- [spaCy](https://spacy.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI GPT](https://platform.openai.com/)
- [Neo4j](https://neo4j.com/)
- [Plotly](https://plotly.com/)
- [Matplotlib](https://matplotlib.org/)
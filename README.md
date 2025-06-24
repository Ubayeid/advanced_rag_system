# Hybrid RAG System: Semantic Retrieval and Knowledge Graph Augmented Generation

## Overview

This project implements a sophisticated Retrieval Augmented Generation (RAG) system that combines vector-based semantic search with a structured Knowledge Graph (KG) for enhanced information retrieval and accurate, context-aware answer generation.

Unlike traditional RAG systems that rely solely on vector similarity, this hybrid approach leverages a Knowledge Graph to:

  * Identify and extract entities and relations from documents.
  * Enrich retrieved content with semantically related information.
  * Improve the relevance and interconnectedness of context provided to the Large Language Model (LLM).

The system is designed for local deployment, offering persistence for its knowledge base components (FAISS index, Knowledge Graph) to avoid re-processing documents on every run unless the source data changes.

## Features

  * **Hybrid Retrieval:** Combines:
      * **Vector Search:** Utilizes Sentence Transformers (default `all-MiniLM-L6-v2`) and FAISS for efficient semantic similarity search on document chunks.
      * **Knowledge Graph Augmentation:** Leverages a NetworkX-based Knowledge Graph to extract entities (using SpaCy's NER and custom rules) and relations, boosting retrieval scores for semantically connected information.
  * **Intelligent Document Processing:**
      * Handles both plain text (`.txt` files) and PDF (`.pdf` files).
      * Employs `pypdf` for robust PDF text extraction.
      * Includes a `_clean_text` function with advanced regex patterns to remove PDF artifacts and noise.
      * Implements a recursive, token-aware chunking strategy with configurable overlap.
      * Extracts keywords using TF-IDF.
  * **Knowledge Graph Construction:**
      * Automatically builds a directed multigraph (`networkx.MultiDiGraph`) of entities, relations, and document chunks.
      * Identifies entities using SpaCy's pre-trained models and custom, hydrology-specific `PhraseMatcher` rules.
      * Extracts relations using SpaCy's `Matcher` with rule-based patterns and falls back to co-occurrence for broader connections.
  * **LLM Integration:** Connects to OpenAI's API (default `gpt-3.5-turbo`) to generate coherent and factual answers based *only* on the retrieved context.
  * **Persistence & Cache:**
      * Automatically saves/loads the FAISS index, KG, and all metadata to/from disk (`./storage` directory).
      * Detects changes in the `./data` directory to trigger full re-processing, otherwise loads the cached knowledge base for quick startup.
  * **Interactive Query Interface:** Provides a simple command-line interface for querying the RAG system.
  * **Knowledge Graph Analysis & Visualization:**
      * Generates a comprehensive Markdown report (`knowledge_gap_report.md`) detailing structural, content, and connectivity gaps in the KG.
      * Creates an interactive Plotly visualization of the Knowledge Graph saved as an HTML file (`knowledge_graph_interactive.html`).

## Future Development

This project is actively under development with the following exciting next steps:

  * **Zotero Database Integration:** The next major step will involve linking the CIROH database (likely managed via Zotero) directly with the project. This will allow for seamless ingestion of research papers and other relevant documents.
  * **User Interface (UI) Development:** A user-friendly graphical interface will be developed to enhance interaction with the RAG system, making it more accessible for users to input queries and visualize results.
  * **Evaluation:** Comprehensive evaluation metrics and methodologies will be implemented to rigorously assess the performance, accuracy, and efficiency of the RAG system.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-project-directory> # e.g., ciroh_new
    ```

2.  **Create a Python Virtual Environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download SpaCy Language Model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up Environment Variables:**
    Create a `.env` file in the root of your project directory (`ciroh_new/`) and add your OpenAI API key and other configurations:

    ```ini
    # .env
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY" # Required for LLM generation
    EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2" # Sentence Transformer model for embeddings
    EMBEDDING_DIM=384 # Dimension for all-MiniLM-L6-v2
    LLM_MODEL_NAME="gpt-3.5-turbo" # LLM for answer generation
    LLM_TEMPERATURE=0.7
    LLM_MAX_TOKENS_CONTEXT=4000
    LLM_MAX_TOKENS_RESPONSE=1000
    TARGET_CHUNK_SIZE_TOKENS=512
    CHUNK_OVERLAP_SENTENCES=2
    MIN_CHUNK_SIZE_TOKENS=20
    STORAGE_PATH="./storage"
    DATA_PATH="./data" # Points to the directory where you'll put your documents
    ENABLE_CACHE="true"
    CACHE_DIR="./cache"
    MAX_CONCURRENT_REQUESTS=10
    REQUEST_TIMEOUT=60
    CACHE_VALIDITY_DAYS=1
    SPACY_MODEL="en_core_web_sm"
    ```

## Usage

### 1\. Prepare Your Data

**The `./data` directory included in this repository serves as a placeholder.** To use the RAG system, you need to add your own source documents (plain text `.txt` files or PDF `.pdf` files) into this `./data` directory. For instance, you could place a research paper on hydrology here.

### 2\. Run the System

The project uses a `Makefile` for convenient execution.

```bash
make run
```

This command will:

  * Activate the virtual environment.
  * Initialize the RAG system.
  * **If your data has changed (or it's the first run):** It will re-process all documents from the `./data` directory, build/update the FAISS index and Knowledge Graph, and save them to the `./storage` directory.
  * **If your data has not changed:** It will quickly load the pre-built knowledge base from `./storage`.
  * Generate the Knowledge Graph analysis report (`knowledge_gap_report.md`) and save it.
  * Generate the interactive Knowledge Graph visualization (`knowledge_graph_interactive.html`) and save it.
  * Start an interactive command-line interface for you to query the system.

### 3\. View the Knowledge Graph Visualization

After running `make run`, you will see messages indicating that the interactive graph HTML file has been saved:

```
Interactive knowledge graph saved to 'knowledge_graph_interactive.html' in your project directory.
Please open this file manually in your web browser.
```

To view the interactive graph:

1.  Open your **Windows File Explorer**.
2.  Navigate to your project's root directory (e.g., `C:\Users\mduba\Development\projects\ai\hybrid\ciroh_new\`).
3.  Double-click on `knowledge_graph_interactive.html`. It will open in your default web browser.

### 4\. View the Knowledge Graph Analysis Report

The detailed markdown report is saved to:

  * `./knowledge_gap_report.md`

You can open this file with any text editor or a Markdown viewer.

### 5\. Query the System

In the terminal where you ran `make run`, you'll see a `Query>` prompt.

```
Enter your query (type 'exit' to quit):
Query> What is an academic paper knowledge graph?
```

Type your questions and press Enter. The system will retrieve relevant information, generate an answer, and provide source attribution and confidence scores.

### 6\. Managing Data Changes

If you modify, add, or remove documents in the `./data` directory, simply run `make run` again. The system will automatically detect the changes (via a hash check of the data directory) and re-process everything, ensuring your knowledge base is always up-to-date.

## Project Structure

```
.
├── main.py                     # Main application logic and RAG system orchestration
├── knowledge_graph.py          # Classes for Knowledge Graph visualization and analysis
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys, configurations)
├── Makefile                    # Automation for running the project
├── data/                       # Directory for raw input documents (PLACE YOUR .txt OR .pdf FILES HERE)
├── storage/                    # Directory for processed/cached knowledge base components
│   ├── faiss_index.bin         # Stored FAISS vector index
│   ├── chunk_metadata.json     # Metadata for all document chunks
│   ├── id_to_index.json        # Mapping of chunk IDs to FAISS indices
│   ├── index_to_id.json        # Mapping of FAISS indices to chunk IDs
│   ├── kg_graph.gml            # NetworkX Knowledge Graph in GML format
│   ├── kg_entities.json        # JSON serialization of KnowledgeEntity objects
│   ├── kg_relations.json       # JSON serialization of KnowledgeRelation objects
│   ├── kg_chunks.json          # JSON serialization of DocumentChunk objects (metadata only)
│   └── data_hash.txt           # Hash of the 'data' directory for change detection
└── # ... other project files ...
```

## Troubleshooting

  * **`ModuleNotFoundError: No module named 'pypdf'` (or other package):**
      * Ensure your virtual environment is activated (`source venv/bin/activate`).
      * Run `pip install -r requirements.txt`.
  * **`SyntaxError` in `knowledge_graph.py`:**
      * Carefully review the exact line number indicated in the traceback. This is usually due to a typo or a duplicated keyword argument (e.g., `x=` or `y=`) in a dictionary or function call.
  * **`Invalid property specified for object of type plotly.graph_objs.Layout: 'titlefont'` (or similar Plotly errors):**
      * Check Plotly's documentation for the correct parameter names. Ensure `titlefont_size` is replaced with `title=dict(font=dict(size=...))`.
  * **`xdg-open: not found` or `no method available for opening 'http://127.0.0.1:XXXX'`:**
      * This means your WSL environment cannot automatically launch a web browser. The HTML file *is* being created successfully.
      * **Manually open** `knowledge_graph_interactive.html` from your Windows File Explorer in the project directory.
      * (Optional) To try and enable `xdg-open` in WSL, you might need to install `xdg-utils`: `sudo apt install xdg-utils`.
  * **"I cannot answer the question based on the given information." for a relevant query:**
      * Verify the content in your `./data` directory actually contains the answer.
      * Check your `_clean_text` function in `main.py` for overly aggressive cleaning that might be removing important content.
      * Temporarily add `logger.info` statements in `HybridRetrievalEngine.retrieve` and `ResponseGenerator.generate_response` to inspect `retrieved_chunks` and `full_context` to see what information the LLM is receiving.
  * **Nonsensical entities/keywords in `Sources Used`:**
      * Refine the `_clean_text` regex patterns in `main.py` to better filter out PDF artifacts (e.g., page numbers, internal metadata, OCR errors).

-----
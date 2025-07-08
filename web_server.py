# web_server.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os
import uuid
import hashlib
import re
import asyncio
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Import core RAG components and constants from main.py
from main import (
    DocumentChunk, KnowledgeEntity, KnowledgeRelation, EmbeddingService,
    DocumentProcessor, VectorIndexManager, KnowledgeGraphManager,
    QueryProcessor, HybridRetrievalEngine, ResponseGenerator, StorageManager,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, LLM_MODEL_NAME, LLM_TEMPERATURE,
    LLM_MAX_TOKENS_CONTEXT, LLM_MAX_TOKENS_RESPONSE, TARGET_CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_SENTENCES, MIN_CHUNK_SIZE_TOKENS, STORAGE_PATH, DATA_PATH,
    ENABLE_CACHE, CACHE_DIR, MAX_CONCURRENT_REQUESTS, REQUEST_TIMEOUT,
    CACHE_VALIDITY_DAYS, SPACY_MODEL, LAST_DATA_HASH_PATH, KG_GRAPH_PATH,
    KG_ENTITIES_PATH, KG_RELATIONS_PATH, KG_CHUNKS_PATH, FAISS_INDEX_PATH,
    CHUNK_METADATA_PATH, ID_TO_INDEX_PATH, INDEX_TO_ID_PATH
)
# DOCUMENT_HASHES_PATH add later
# Import networkx for graph manipulation (NEW ADDITION)
import networkx as nx

# Import knowledge_graph for visualization if enabled
import knowledge_graph

# Define GENERATE_KG_VISUALIZATION here as it's used in SimpleAdvancedRAGSystem
GENERATE_KG_VISUALIZATION = os.getenv("GENERATE_KG_VISUALIZATION", "true").lower() == "true"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# MAIN RAG SYSTEM
# =============================================================================
class SimpleAdvancedRAGSystem:
    def __init__(self):
        # Ensure paths exist before initializing components that use them
        STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.embedding_service = EmbeddingService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIM
        )

        self.storage_manager = StorageManager(DATA_PATH, STORAGE_PATH)
        self.doc_processor = DocumentProcessor(self.embedding_service)
        self.vector_index = VectorIndexManager(embedding_dim=EMBEDDING_DIM)
        self.kg_manager = KnowledgeGraphManager()
        self.retrieval_engine = HybridRetrievalEngine(self.vector_index, self.kg_manager, self.embedding_service)
        self.response_generator = ResponseGenerator()

        logger.info("RAG System components initialized successfully.")

    def initialize_knowledge_base(self):
        """Initializes or re-processes the knowledge base based on data changes."""
        needs_reprocessing = self.storage_manager.is_data_changed() # Determine if reprocessing is needed

        # Attempt to load existing data
        if self.storage_manager.load_all(self.vector_index, self.kg_manager) and not needs_reprocessing:
            logger.info("Knowledge base is up-to-date. No re-processing needed.")
            return

        logger.info("Re-processing documents due to data changes or no prior data found.")

        self.vector_index._initialize_index()
        self.vector_index.chunk_metadata = {}
        self.vector_index.id_to_index = {}
        self.vector_index.index_to_id = {}
        self.vector_index.current_index = 0

        self.kg_manager.graph = nx.MultiDiGraph()
        self.kg_manager.entities = {}
        self.kg_manager.relations = {}
        self.kg_manager.document_chunks = {}

        # Perform the processing and saving
        self._process_and_store_documents()
        self.storage_manager.save_all(self.vector_index, self.kg_manager)
        logger.info("Knowledge base re-processing complete and saved.")

        if GENERATE_KG_VISUALIZATION:
            logger.info("Generating interactive knowledge graph visualization ...")
            try:
                knowledge_graph.analyze_rag_knowledge_graph(self)

                visualizer = knowledge_graph.KnowledgeGraphVisualizer(self.kg_manager)
                
                static_graph_save_path = os.path.join(os.getenv("STORAGE_PATH", "./storage"), "knowledge_graph_static.png")
                
                visualizer.visualize_graph_static(save_path=static_graph_save_path)
                logger.info(f"Generating static knowledge graph visualization ...")

            except Exception as e:
                logger.error(f"Error during knowledge graph visualization/analysis (after reprocessing): {e}")
        else:
            logger.info("Skipping knowledge graph visualization and gap report generation as per configuration (GENERATE_KG_VISUALIZATION is false).")

    def _process_and_store_documents(self):
        """Processes documents from the DATA_PATH and stores them."""
        documents = self._load_documents_from_directory(DATA_PATH)

        if not documents:
            logger.warning(f"No documents found in {DATA_PATH}. Knowledge base will be empty.")
            return

        all_chunks = []
        all_entities = []
        all_relations = []

        for doc in documents:
            doc_id = doc['id']
            content = doc['content']

            # Process document
            chunks, entities, relations = self.doc_processor.process_document(doc_id, content)

            all_chunks.extend(chunks)
            all_entities.extend(entities)
            all_relations.extend(relations)

        self.vector_index.add_chunks(all_chunks)
        self.kg_manager.add_data(all_chunks, all_entities, all_relations)

        logger.info(f"Processed {len(documents)} documents, Created {len(all_chunks)} chunks, {len(all_entities)} Entities, {len(all_relations)} Relations.")

    def _load_documents_from_directory(self, directory: Path) -> List[Dict[str, str]]:
        """Load documents from a specified directory."""
        documents = []
        # Import pypdf locally to avoid import errors if it's not strictly needed elsewhere
        import pypdf
        for file_path in directory.iterdir():
            if file_path.is_file():
                content = ""
                if file_path.suffix.lower() == '.pdf':
                    try:
                        reader = pypdf.PdfReader(file_path)
                        for page in reader.pages:
                            content += page.extract_text(0) + "\n"
                        logger.info(f"Loaded PDF document: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Could not read PDF {file_path} using pypdf: {e}")
                        continue
                elif file_path.suffix.lower() == '.txt':
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        logger.info(f"Loaded TXT document: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Could not read TXT {file_path}: {e}")
                        continue
                else:
                    logger.warning(f"Unsupported file type: {file_path.suffix} for {file_path.name}. Skipping.")
                    continue

                if content:
                    documents.append({'id': file_path.name, 'content': content})
        return documents

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query and return results."""
        logger.info(f"Processing query: '{query_text}'")
        try:
            retrieved_chunks = self.retrieval_engine.retrieve(query_text, top_k)
            query_analysis = self.retrieval_engine.query_processor.analyze_query(query_text)
            response = self.response_generator.generate_response(
                query_text, retrieved_chunks, query_analysis
            )
            response['query_analysis'] = query_analysis
            response['timestamp'] = datetime.now().isoformat()
            response['status'] = 'success'
            logger.info("Query processed successfully.")
            return response
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return {
                'status': 'success',
                'vector_index_size': self.vector_index.index.ntotal if self.vector_index.index else 0,
                'knowledge_graph_nodes': self.kg_manager.graph.number_of_nodes(),
                'knowledge_graph_edges': self.kg_manager.graph.number_of_edges(),
                'total_entities': len(self.kg_manager.entities),
                'total_relations': len(self.kg_manager.relations),
                'total_chunks_in_kg': len(self.kg_manager.document_chunks),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global instance of the RAG system
rag_system_instance = None

try:
    rag_system_instance = SimpleAdvancedRAGSystem()
    rag_system_instance.initialize_knowledge_base()
except Exception as e:
    logger.error(f"Failed to initialize RAG System on startup: {e}")
    rag_system_instance = None

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/query', methods=['POST'])
def handle_query():
    if rag_system_instance is None:
        return jsonify({"status": "error", "message": "RAG system not initialized. Check server logs for errors."}), 500

    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"status": "error", "message": "Query parameter is missing."}), 400

    logger.info(f"Received query: {user_query}")
    response_data = rag_system_instance.query(user_query)
    return jsonify(response_data)

@app.route('/stats', methods=['GET'])
def get_stats():
    if rag_system_instance is None:
        return jsonify({"status": "error", "message": "RAG system not initialized."}), 500
    stats = rag_system_instance.get_system_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
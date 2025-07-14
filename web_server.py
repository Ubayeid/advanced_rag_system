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
    QueryProcessor, HybridRetrievalEngine, ResponseGenerator,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, LLM_MODEL_NAME, LLM_TEMPERATURE,
    LLM_MAX_TOKENS_CONTEXT, LLM_MAX_TOKENS_RESPONSE, TARGET_CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_SENTENCES, MIN_CHUNK_SIZE_TOKENS, STORAGE_PATH, DATA_PATH,
    ENABLE_CACHE, CACHE_DIR, MAX_CONCURRENT_REQUESTS, REQUEST_TIMEOUT,
    CACHE_VALIDITY_DAYS, SPACY_MODEL, FAISS_INDEX_PATH, LAST_DATA_HASH_PATH,
    CHUNK_METADATA_PATH, ID_TO_INDEX_PATH, INDEX_TO_ID_PATH, extract_document_level_metadata
)
# DOCUMENT_HASHES_PATH add later
# Import networkx for graph manipulation (NEW ADDITION)
import networkx as nx

# Import knowledge_graph specifically for the analyze_rag_knowledge_graph function
import knowledge_graph

# Define GENERATE_KG_VISUALIZATION here as it's used in SimpleAdvancedRAGSystem
GENERATE_KG_VISUALIZATION = os.getenv("GENERATE_KG_VISUALIZATION", "false").lower() == "true" # Set to false by default as per request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class StorageManager:

    def __init__(self, data_path: Path, storage_path: Path):
        self.data_path = data_path
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _calculate_data_hash(self) -> str:
        """Calculates a hash of all files in the data directory."""
        hasher = hashlib.md5()
        file_paths = sorted(list(self.data_path.rglob('*')))
        for file_path in file_paths:
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            hasher.update(chunk)
                except Exception as e:
                    logger.error(f"Error reading file {file_path} for hash calculation: {e}")
                    continue
        return hasher.hexdigest()

    def _get_last_data_hash(self) -> Optional[str]:
        """Reads the last stored data hash."""
        if LAST_DATA_HASH_PATH.exists():
            try:
                return LAST_DATA_HASH_PATH.read_text().strip()
            except Exception as e:
                logger.error(f"Error reading last data hash from {LAST_DATA_HASH_PATH}: {e}")
                return None
        return None

    def _save_data_hash(self, current_hash: str) -> bool:
        """Saves the current data hash."""
        try:
            LAST_DATA_HASH_PATH.write_text(current_hash)
            return True
        except Exception as e:
            logger.error(f"Error saving data hash to {LAST_DATA_HASH_PATH}: {e}")
            return False

    def is_data_changed(self) -> bool:
        """Checks if the data directory has changed since last processing."""
        current_hash = self._calculate_data_hash()
        last_hash = self._get_last_data_hash()

        if current_hash != last_hash:
            logger.info("Data directory has changed or no previous hash found. Re-processing required.")
            return True
        logger.info("Data directory is unchanged. Loading pre-calculated data.")
        return False

    def save_all(self, vector_index_manager: VectorIndexManager, kg_manager: KnowledgeGraphManager):
        """Saves all relevant data structures. Only saves hash if all saves succeed."""
        all_saves_successful = True

        if not vector_index_manager.save_index():
            all_saves_successful = False
            logger.error("Failed to save FAISS index. Skipping data hash update.")

        if not kg_manager.save_graph():
            all_saves_successful = False
            logger.error("Failed to save Knowledge Graph. Skipping data hash update.")

        if all_saves_successful:
            if self._save_data_hash(self._calculate_data_hash()):
                logger.info("All data structures and data hash saved successfully.")
            else:
                logger.error("Failed to save data hash after successful index/graph saves. Data may be inconsistent on next load.")
        else:
            logger.error("Some data structures failed to save. Data hash not updated.")

    def load_all(self, vector_index_manager: VectorIndexManager, kg_manager: KnowledgeGraphManager) -> bool:
        """Loads all relevant data structures. Returns True if successful, False otherwise."""
        vector_loaded = vector_index_manager.load_index()
        kg_loaded = kg_manager.load_graph()

        if vector_loaded and kg_loaded and LAST_DATA_HASH_PATH.exists():
            # Only consider "loaded" if data hash exists, implies previous full save
            logger.info("All data structures loaded successfully.")
            return True

        logger.warning("Incomplete or failed load of data structures. Will re-process data.")
        return False

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
        self.vector_index = VectorIndexManager(embedding_dim=EMBEDDING_DIM) # Initial creation
        self.kg_manager = KnowledgeGraphManager()
        self.retrieval_engine = HybridRetrievalEngine(self.vector_index, self.kg_manager, self.embedding_service)
        self.response_generator = ResponseGenerator(kg_manager=self.kg_manager)

        logger.info("RAG System components initialized successfully.")

    def initialize_knowledge_base(self):
        """Initializes or re-processes the knowledge base based on data changes."""
        needs_reprocessing = self.storage_manager.is_data_changed() # Determine if reprocessing is needed

        # Attempt to load existing data
        if self.storage_manager.load_all(self.vector_index, self.kg_manager) and not needs_reprocessing:
            logger.info("Knowledge base is up-to-date. No re-processing needed.")
            # Still generate report even if not reprocessing data, if desired
            logger.info("Generating comprehensive gap analysis report with research hypotheses...")
            try:
                # Call the enhanced analyze_rag_knowledge_graph function from knowledge_graph.py
                knowledge_graph.analyze_rag_knowledge_graph(self)
            except Exception as e:
                logger.error(f"Error generating enhanced knowledge gap report: {e}")
            return

        logger.info("Re-processing documents due to data changes or no prior data found.")

        # Correct fix: Re-instantiate VectorIndexManager for a completely fresh FAISS index
        self.vector_index = VectorIndexManager(embedding_dim=EMBEDDING_DIM) 
        # Crucial: Update retrieval_engine with the new vector_index instance
        self.retrieval_engine.vector_index = self.vector_index

        # Reset KG Manager's in-memory caches, but not the Neo4j driver itself
        # For Neo4j, simply ensure the database is clear if reprocessing from scratch
        if self.kg_manager.driver:
            try:
                with self.kg_manager.driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n") # Clear entire graph
                logger.info("Neo4j database cleared for full reprocessing.")
            except Exception as e:
                logger.error(f"Failed to clear Neo4j database: {e}. Continuing with existing data if any.")

        self.kg_manager.entities = {}
        self.kg_manager.relations = [] # This is a transient list, can be safely reset
        self.kg_manager.document_chunks = {}
        self.kg_manager.all_document_metadata = {} # Reset document metadata on reprocessing

        # Perform the processing and saving
        self._process_and_store_documents()
        self.storage_manager.save_all(self.vector_index, self.kg_manager)
        logger.info("Knowledge base re-processing complete and saved.")

        # Generate the gap report after processing and saving
        logger.info("Generating comprehensive gap analysis report with research hypotheses...")
        try:
            # Call the enhanced analyze_rag_knowledge_graph function from knowledge_graph.py
            knowledge_graph.analyze_rag_knowledge_graph(self)
        except Exception as e:
            logger.error(f"Error generating enhanced knowledge gap report: {e}")

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

            # Extract document-level metadata (like authors and title)
            doc_metadata = extract_document_level_metadata(content, doc_id)
            self.kg_manager.add_document_level_metadata(doc_id, doc_metadata)

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
        import pypdf # Import pypdf locally to avoid import errors if it's not strictly needed elsewhere
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

        kg_node_count = 0
        kg_edge_count = 0
        if self.kg_manager.driver:
            try:
                with self.kg_manager.driver.session() as session:
                    count_result = session.run("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS edges").single()
                    kg_node_count = count_result['nodes']
                    kg_edge_count = count_result['edges']
            except Exception as e:
                logger.warning(f"Could not fetch live Neo4j stats: {e}")

        return {
            'status': 'success',
            'vector_index_size': self.vector_index.index.ntotal if self.vector_index.index else 0,
            'knowledge_graph_nodes': kg_node_count, # Dynamically fetched
            'knowledge_graph_edges': kg_edge_count, # Dynamically fetched
            'total_entities': len(self.kg_manager.entities), # From in-memory cache
            'total_relations': len(self.kg_manager.relations), # From in-memory cache (note: this is a transient list from processing, not live Neo4j count)
            'total_chunks_in_kg': len(self.kg_manager.document_chunks), # From in-memory cache
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
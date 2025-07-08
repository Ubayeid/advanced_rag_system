import pypdf
import asyncio
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import uuid
import logging
import re
import os
import argparse
from dotenv import load_dotenv
import sys
import hashlib
import tiktoken

# Core libraries
import faiss
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.matcher import Matcher, PhraseMatcher

# OpenAI & Sentence Transformers
from openai import OpenAI, APIError
from sentence_transformers import SentenceTransformer

import knowledge_graph

# Plotly specific import for renderer control
import plotly.io as pio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Configuration Constants (Loaded from .env or default values)
# =============================================================================

# Embedding Model Configuration (for local embeddings via Sentence Transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") # Default to ST model
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384")) # Default dimension for all-MiniLM-L6-v2

# LLM Model Configuration (for answer generation via OpenAI API)
LLM_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS_CONTEXT = int(os.getenv("LLM_MAX_TOKENS_CONTEXT", "4000")) # Max input tokens for LLM
LLM_MAX_TOKENS_RESPONSE = int(os.getenv("LLM_MAX_TOKENS_RESPONSE", "1000")) # Max output tokens for LLM response

# Document Processing Settings (using token-based chunking for local models)
TARGET_CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "512")) # Max tokens per chunk for embeddings
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2")) # Overlap in sentences
MIN_CHUNK_SIZE_TOKENS = int(os.getenv("MIN_CHUNK_SIZE_TOKENS", "20")) # Minimum tokens for a chunk

# Storage Paths
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./storage"))
DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))

# --- Persistence Paths (derived from STORAGE_PATH) ---
FAISS_INDEX_PATH = STORAGE_PATH / "faiss_index.bin"
CHUNK_METADATA_PATH = STORAGE_PATH / "chunk_metadata.json"
ID_TO_INDEX_PATH = STORAGE_PATH / "id_to_index.json"
INDEX_TO_ID_PATH = STORAGE_PATH / "index_to_id.json"
KG_GRAPH_PATH = STORAGE_PATH / "kg_graph.gml"
KG_ENTITIES_PATH = STORAGE_PATH / "kg_entities.json"
KG_RELATIONS_PATH = STORAGE_PATH / "kg_relations.json"
LAST_DATA_HASH_PATH = STORAGE_PATH / "data_hash.txt"
KG_CHUNKS_PATH = STORAGE_PATH / "kg_chunks.json"
# DOCUMENT_HASHES_PATH = STORAGE_PATH / "document_hashes.json"
# System Settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
CACHE_VALIDITY_DAYS = int(os.getenv("CACHE_VALIDITY_DAYS", "1"))

# SpaCy Model Configuration
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Ensure storage and data directory exist
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class DocumentChunk:
    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    page_number: Optional[int] = None

    def to_dict(self):
        return {
            'id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'chunk_index': self.chunk_index,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'keywords': self.keywords,
            'entities': self.entities,
            'metadata': self.metadata,
            'section_title': self.section_title,
            'subsection_title': self.subsection_title,
            'page_number': self.page_number
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return DocumentChunk(
            id=data['id'],
            document_id=data['document_id'],
            content=data['content'],
            chunk_index=data['chunk_index'],
            embedding=np.array(data['embedding']).astype(np.float32) if data['embedding'] is not None else None,
            keywords=data['keywords'],
            entities=data['entities'],
            metadata=data['metadata'],
            section_title=data.get('section_title'),
            subsection_title=data.get('subsection_title'),
            page_number=data.get('page_number')
        )

@dataclass
class KnowledgeEntity:
    id: str
    name: str
    entity_type: str
    confidence: float
    document_ids: List[str] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return KnowledgeEntity(**data)

@dataclass
class KnowledgeRelation:
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    document_id: Optional[str] = None
    sentence: Optional[str] = None

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return KnowledgeRelation(**data)


# 2. UTILITY SERVICES (Embedding, etc.)
# =============================================================================

class EmbeddingService:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        if model_name.startswith("text-embedding-") or model_name.startswith("gpt-"):
            if not api_key:
                raise ValueError(f"OPENAI_API_KEY must be set for OpenAI embedding model: {model_name}")
            self.client = OpenAI(api_key=api_key)
            self.model_type = "openai_api"
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            self.EMBEDDING_MODEL_MAX_TOKENS = 8192 # OpenAI Ada-002 limit
            logger.info(f"EmbeddingService initialized with OpenAI API model: {model_name}")
        else:
            try:
                self.model = SentenceTransformer(model_name)
                self.model_type = "sentence_transformer"
                self.tokenizer = tiktoken.encoding_for_model(LLM_MODEL_NAME)
                self.EMBEDDING_MODEL_MAX_TOKENS = self.model.max_seq_length
                if self.EMBEDDING_MODEL_MAX_TOKENS is None:
                    self.EMBEDDING_MODEL_MAX_TOKENS = 512
                    logger.warning(f"SentenceTransformer model '{model_name}' did not expose max_seq_length. Defaulting to {self.EMBEDDING_MODEL_MAX_TOKENS} tokens for internal chunking checks.")
                logger.info(f"EmbeddingService initialized with local Sentence Transformer model: {model_name} (Max Seq Len: {self.EMBEDDING_MODEL_MAX_TOKENS})")
            except Exception as e:
                logger.error(f"Failed to load Sentence Transformer model '{model_name}'. Ensure it's valid and downloaded (e.g., pip install sentence-transformers, then model name is correct in .env). Error: {e}")
                raise RuntimeError(f"Failed to load embedding model: {model_name}")

    def get_embedding(self, text: str, retries: int = 3, delay: int = 1) -> np.ndarray:
        """Generates embedding for a given text using the configured model."""
        if self.model_type == "openai_api":
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.EMBEDDING_MODEL_MAX_TOKENS:
                logger.error(f"Text too long for OpenAI embedding model ({token_count} tokens). Max is {self.EMBEDDING_MODEL_MAX_TOKENS}. Snippet: {text[:200]}...")
                return np.zeros(self.embedding_dim, dtype=np.float32)

            for i in range(retries):
                try:
                    response = self.client.embeddings.create(model=self.model_name, input=text)
                    return np.array(response.data[0].embedding).astype(np.float32)
                except APIError as e:
                    if e.code == 'invalid_request_error' and "context length" in str(e):
                        logger.error(f"OpenAI API confirmed token limit error ({token_count} tokens) for text. Snippet: {text[:200]}... Error: {e}")
                        return np.zeros(self.embedding_dim, dtype=np.float32)

                    logger.warning(f"OpenAI API error during embedding (attempt {i+1}/{retries}): {e}")
                    if i < retries - 1:
                        import time
                        time.sleep(delay * (2 ** i))
                    else:
                        logger.error(f"Failed to get embedding after {retries} attempts for text: {text[:50]}... Final error: {e}")
                        return np.zeros(self.embedding_dim, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Unexpected error getting OpenAI embedding for text: {text[:50]}... Error: {e}")
                    return np.zeros(self.embedding_dim, dtype=np.float32)
            return np.zeros(self.embedding_dim, dtype=np.float32)

        elif self.model_type == "sentence_transformer":
            try:
                embedding = self.model.encode(text, convert_to_numpy=True).astype(np.float32)
                if embedding.shape[0] != self.embedding_dim:
                    logger.warning(f"SentenceTransformer embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}. Check EMBEDDING_DIM in .env for {self.model_name}. Attempting to reshape or pad.")
                    if embedding.shape[0] < self.embedding_dim:
                        padded_embedding = np.pad(embedding, (0, self.embedding_dim - embedding.shape[0]), 'constant')
                        return padded_embedding.astype(np.float32)
                    elif embedding.shape[0] > self.embedding_dim:
                        return embedding[:self.embedding_dim].astype(np.float32)
                return embedding
            except Exception as e:
                logger.error(f"Error generating Sentence Transformer embedding for text: {text[:50]}... Error: {e}")
                return np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            logger.error(f"Unknown embedding model type: {self.model_type}")
            return np.zeros(self.embedding_dim, dtype=np.float32)


# 3. DOCUMENT PROCESSOR
# =============================================================================

class DocumentProcessor:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.tokenizer = embedding_service.tokenizer

        try:
            self.nlp = spacy.load(SPACY_MODEL)
            self.nlp.max_length = 2_000_000
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab)

            # --- 1. Define Custom Hydrology Entity Types (Rule-Based) ---
            water_body_terms = [
                "river", "lake", "stream", "aquifer", "reservoir", "ocean", "sea",
                "wetland", "marsh", "swamp", "estuary", "fjord", "bay", "lagoon",
                "groundwater", "surface water", "drinking water", "dam", "canal",
                "watershed", "basin", "drainage basin", "catchment area", "glacier", "ice cap"
            ]
            self.phrase_matcher.add("WATER_BODY", [self.nlp(text) for text in water_body_terms])

            measurement_terms = [
                "flow rate", "water level", "stage height", "precipitation", "rainfall", "snowfall",
                "temperature", "salinity", "pH", "dissolved oxygen", "turbidity", "conductivity",
                "discharge", "volume", "depth", "velocity", "humidity", "evaporation", "transpiration"
            ]
            self.phrase_matcher.add("HYDRO_MEASUREMENT", [self.nlp(text) for text in measurement_terms])

            pollutant_terms = [
                "nitrate", "phosphate", "sulfate", "mercury", "lead", "cadmium", "arsenic",
                "pesticide", "herbicide", "heavy metal", "microplastic", "E. coli",
                "contaminant", "pollutant", "nutrient", "algae"
            ]
            self.phrase_matcher.add("POLLUTANT", [self.nlp(text) for text in pollutant_terms])

            event_terms = [
                "flood", "drought", "storm", "hurricane", "typhoon", "cyclone",
                "runoff", "infiltration", "percolation", "erosion", "sedimentation",
                "desalination", "recharge", "effluent", "discharge event"
            ]
            self.phrase_matcher.add("HYDRO_EVENT", [self.nlp(text) for text in event_terms])

            infrastructure_terms = [
                "gauge", "sensor", "monitor", "weather station", "well", "pump",
                "treatment plant", "wastewater facility", "dam", "levee", "dyke", "pipe", "culvert"
            ]
            self.phrase_matcher.add("HYDRO_INFRASTRUCTURE", [self.nlp(text) for text in infrastructure_terms])

            model_terms = [
                "SWMM", "HEC-RAS", "MODFLOW", "VIC model", "WRF-Hydro", "WEAP"
            ]
            self.phrase_matcher.add("HYDRO_MODEL", [self.nlp(text) for text in model_terms])

            study_terms = [
                "study", "research", "project", "analysis", "investigation", "report", "assessment"
            ]
            self.phrase_matcher.add("STUDY", [self.nlp(text) for text in study_terms])

            dataset_terms = [
                "IBTRACS", "ERA5", "NOAA OISST", "OISST", "Copernicus Climate Data Store",
                "NCEI", "International Best Track Archive for Climate Stewardship", "IBTrACS"
            ]
            self.phrase_matcher.add("DATASET", [self.nlp(text) for text in dataset_terms])

            # --- 2. Define Custom Hydrology Relation Patterns ---
            self.matcher.add("FLOWS_INTO", [[
                {"ENT_TYPE": {"IN": ["WATER_BODY", "GPE"]}},
                {"LEMMA": {"IN": ["flow", "drain", "empty", "discharge"]}},
                {"POS": "ADP", "OP": "*"},
                {"ENT_TYPE": {"IN": ["WATER_BODY", "GPE"]}}
            ]])

            self.matcher.add("MEASURES", [[
                {"ENT_TYPE": "HYDRO_INFRASTRUCTURE"},
                {"LEMMA": {"IN": ["measure", "detect", "record", "monitor", "gauge"]}},
                {"ENT_TYPE": {"IN": ["HYDRO_MEASUREMENT", "POLLUTANT"]}}
            ]])

            self.matcher.add("CONTAINS", [[
                {"ENT_TYPE": "WATER_BODY"},
                {"LEMMA": {"IN": ["contain", "have", "include", "exhibit"]}},
                {"ENT_TYPE": "POLLUTANT"}
            ]])

            self.matcher.add("CAUSES", [[
                {"ENT_TYPE": "HYDRO_EVENT"},
                {"LEMMA": "cause"},
                {"ENT_TYPE": {"IN": ["HYDRO_EVENT", "HYDRO_MEASUREMENT"]}}
            ]])

            self.matcher.add("USES_MODEL", [[
                {"ENT_TYPE": {"IN": ["ORG", "PERSON"]}},
                {"LEMMA": {"IN": ["use", "apply", "employ", "utilize", "develop"]}},
                {"ENT_TYPE": "HYDRO_MODEL"}
            ]])

            self.matcher.add("INVESTIGATES", [[
                {"ENT_TYPE": "STUDY"},
                {"LEMMA": {"IN": ["investigate", "analyze", "examine", "assess"]}},
                {"ENT_TYPE": {"IN": ["WATER_BODY", "HYDRO_EVENT", "POLLUTANT", "HYDRO_MEASUREMENT"]}}
            ]])

            self.matcher.add("ENSURES_INDEPENDENCE", [
                [{"LOWER": "minimum"}, {"LOWER": "separation"}, {"LOWER": "time"},
                 {"TEXT": "(", "OP": "?"}, {"TEXT": "MST", "OP": "?"}, {"TEXT": ")", "OP": "?"}],
                [{"LOWER": "declustering"}, {"LOWER": "scheme"}, {"LEMMA": "utilize"}]
            ])

            self.matcher.add("DEFINES_THRESHOLD", [
                [{"LEMMA": "define"}, {"POS": "ADP", "OP": "*"}, {"LOWER": "marine"}, {"LOWER": "heatwave"}],
                [{"LEMMA": "set"}, {"POS": "DET", "OP": "*"}, {"LOWER": "threshold"}]
            ])

            self.matcher.add("QUANTIFIES_IMPACT", [
                [{"LOWER": "multiplication"}, {"LOWER": "rate"}],
                [{"LEMMA": "quantify"}, {"POS": "DET", "OP": "*"}, {"LOWER": "impact"}],
                [{"LOWER": "ratio"}, {"POS": "ADP", "OP": "*"}, {"LOWER": "probabilities"}]
            ])

            self.matcher.add("USES_DATASET", [
                [{"LEMMA": {"IN": ["use", "employ", "leverage", "utilize", "source", "obtain"]}}, {"POS": "DET", "OP": "*"}, {"ENT_TYPE": "DATASET"}],
                [{"LEMMA": {"IN": ["use", "employ", "leverage", "utilize", "source", "obtain"]}}, {"POS": "DET", "OP": "*"}, {"ENT_TYPE": {"IN": ["ORG", "PRODUCT"]}}],
                [{"TEXT": {"IN": ["IBTRACS", "ERA5", "NOAA", "OISST", "Copernicus"]}, "OP": "+"}, {"POS": "NOUN", "OP": "*", "LEMMA": {"IN": ["data", "model", "dataset", "product", "suite", "information"]}}]
            ])

        except OSError:
            logger.error(f"spaCy model '{SPACY_MODEL}' not found. Please install with: python -m spacy download {SPACY_MODEL}")
            self.nlp = None
            self.matcher = None
            self.phrase_matcher = None

        self.target_chunk_size_tokens = TARGET_CHUNK_SIZE_TOKENS
        self.chunk_overlap_sentences = CHUNK_OVERLAP_SENTENCES
        self.min_chunk_size_tokens = MIN_CHUNK_SIZE_TOKENS

        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')

    def process_document(self, document_id: str, content: str) -> Tuple[List[DocumentChunk], List[KnowledgeEntity], List[KnowledgeRelation]]:
        """Process document into chunks with entities and relations, inferring metadata."""

        logger.info(f"Processing document: {document_id}")
        cleaned_content = self._clean_text(content)

        # Extract entities first
        entities = self._extract_entities(cleaned_content, document_id)

        # MODIFIED: Use the new chunking method that infers metadata
        chunks = self._create_chunks_with_metadata_inference(cleaned_content, document_id)

        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = self.embedding_service.get_embedding(chunk.content)

        # Extract keywords
        self._extract_keywords(chunks)

        # Link entities to chunks
        self._link_entities_to_chunks(chunks, entities)

        relations, entities = self._create_relations(entities, cleaned_content, document_id, chunks)

        return chunks, entities, relations

    def _clean_text(self, content: str) -> str:
        return content.strip()

    def _create_chunks_with_metadata_inference(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """
        Creates overlapping chunks, attempting to infer section/page metadata from the document's raw content, especially for PDF-derived text.
        """
        all_final_chunks = [] # Stores all chunks from the entire document
        global_chunk_index = 0 # Unique index across all chunks in the document

        EMBEDDING_MODEL_HARD_MAX_TOKENS = self.embedding_service.EMBEDDING_MODEL_MAX_TOKENS
        FALLBACK_CHAR_CHUNK_SIZE = int(EMBEDDING_MODEL_HARD_MAX_TOKENS * 0.9)

        # Split content by explicit page markers
        # The pattern (r'\n\s*--- PAGE (\d+) ---\s*\n') means:
        # - Group 1 captures the page number.
        # - The split result will be [pre-page-1-content, page-1-number, page-1-content, page-2-number, page-2-content, ...]
        page_splits = re.split(r'\n\s*--- PAGE (\d+) ---\s*\n', content)

        # Process content before the first page marker (if any)
        if page_splits[0].strip():
            # Initial metadata for content before first page marker
            current_section_title = "Document Introduction"
            current_subsection_title = None
            current_page_number = 0 # Or 1, depending on how you want to number pre-page content

            chunks_from_segment = self._process_single_content_segment(
                doc_id, page_splits[0],
                current_section_title, current_subsection_title, current_page_number,
                EMBEDDING_MODEL_HARD_MAX_TOKENS, FALLBACK_CHAR_CHUNK_SIZE,
                global_chunk_index # Pass starting chunk index
            )
            all_final_chunks.extend(chunks_from_segment)
            if chunks_from_segment:
                global_chunk_index = chunks_from_segment[-1].chunk_index + 1

        # Iterate through detected pages and their content (page_splits[1] is page number, page_splits[2] is content, etc.)
        for i in range(1, len(page_splits), 2):
            page_num_str = page_splits[i].strip()
            page_content = page_splits[i+1].strip() # Get content for THIS page

            try:
                current_page_number = int(page_num_str)
            except ValueError:
                logger.warning(f"Could not parse page number from '{page_num_str}' in {doc_id}. Keeping previous page number for safety.")
                # If cannot parse, maybe use the last known page number or a default.
                # For robustness, let's ensure current_page_number is always an int
                current_page_number = current_page_number # Retain previous, or set a default like 1

            if not page_content: # Skip empty page content
                continue

            # Reset section/subsection titles for each new page unless explicitly carried over
            # For simplicity, assuming new page potentially means new section starts unless context dictates otherwise
            current_section_title = "Page Content" # Default for a new page
            current_subsection_title = None

            lines = page_content.split('\n')
            
            # Re-evaluate section/subsection headers within each page
            processed_lines = []
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue # Skip empty lines

                # Heuristic for main section headers (e.g., "1. DATA STRUCTURES")
                # This regex is quite broad, might need refinement
                if re.fullmatch(r'([A-Z\s&]+|\d+\.\s[A-Z\s&]+)', stripped_line) and len(stripped_line) < 50:
                    current_section_title = stripped_line
                    current_subsection_title = None
                # Heuristic for subsection headers (e.g., "RI definition and detection")
                elif re.fullmatch(r'[\w\s&-]+', stripped_line) and stripped_line.istitle() and len(stripped_line.split()) < 8 and len(stripped_line) < 80:
                    # Blacklist common non-headers from your PDF example
                    if stripped_line.lower() not in ["article", "check for updates", "summary", "figure 1", "figure 2", "figure 3", "figure 4", "figure 5", "figure 6", "figure 7", "figure 8", "figure 9"]:
                        current_subsection_title = stripped_line
                
                processed_lines.append(line) # Use original line to preserve formatting for chunking

            # Process this page's content
            chunks_from_segment = self._process_single_content_segment(
                doc_id, "\n".join(processed_lines),
                current_section_title, current_subsection_title, current_page_number,
                EMBEDDING_MODEL_HARD_MAX_TOKENS, FALLBACK_CHAR_CHUNK_SIZE,
                global_chunk_index # Pass starting chunk index
            )
            all_final_chunks.extend(chunks_from_segment)
            if chunks_from_segment:
                global_chunk_index = chunks_from_segment[-1].chunk_index + 1

        logger.info(f"Document {doc_id} split into {len(all_final_chunks)} chunks with inferred metadata.")
        return all_final_chunks

    def _process_single_content_segment(self, doc_id: str, segment_content: str,
                                        section_title: str, subsection_title: Optional[str], page_number: Optional[int],
                                        max_tokens: int, fallback_char_size: int,
                                        start_chunk_index: int) -> List[DocumentChunk]:
        """
        Helper method to process a single continuous content segment (e.g., a page or a pre-page block)
        into overlapping chunks, maintaining a consistent metadata context.
        """
        segment_chunks = []

        pre_overlap_chunks = self._recursive_split_for_metadata_helper(
            segment_content,
            ["\n\n\n", "\n\n", ". ", "? ", "! ", " "],
            max_tokens,
            fallback_char_size
        )

        current_overlap_sentences_buffer = []
        current_combined_chunk_content = []
        current_combined_chunk_tokens = 0
        chunk_idx_for_segment = start_chunk_index

        for pre_chunk_content in pre_overlap_chunks:
            if not pre_chunk_content.strip():
                continue

            if self.nlp:
                doc_pre_chunk = self.nlp(pre_chunk_content)
                sentences_from_pre_chunk = [sent.text.strip() for sent in doc_pre_chunk.sents if sent.text.strip()]
            else:
                sentences_from_pre_chunk = [s.strip() for s in re.split(r'(?<=[.?!])\s+', pre_chunk_content) if s.strip()]

            for sent in sentences_from_pre_chunk:
                if not sent.strip(): # Skip empty sentences
                    continue

                sent_tokens = len(self.tokenizer.encode(sent))

                # Check if adding this sentence exceeds target chunk size
                if current_combined_chunk_tokens + sent_tokens > self.target_chunk_size_tokens:
                    if current_combined_chunk_content: # Ensure there's content to save
                        final_chunk_text = " ".join(current_combined_chunk_content)
                        final_chunk_tokens = len(self.tokenizer.encode(final_chunk_text))

                        if final_chunk_tokens >= self.min_chunk_size_tokens:
                            segment_chunks.append(DocumentChunk(
                                id=str(uuid.uuid4()),
                                document_id=doc_id,
                                content=final_chunk_text,
                                chunk_index=chunk_idx_for_segment,
                                section_title=section_title,
                                subsection_title=subsection_title,
                                page_number=page_number
                            ))
                            chunk_idx_for_segment += 1
                        else:
                            logger.debug(f"Skipping small chunk {chunk_idx_for_segment} from {doc_id}: {final_chunk_tokens} tokens.")

                    # Start new chunk with overlap and current sentence
                    current_combined_chunk_content = list(current_overlap_sentences_buffer)
                    current_combined_chunk_content.append(sent)
                    current_combined_chunk_tokens = len(self.tokenizer.encode(" ".join(current_combined_chunk_content)))
                    current_overlap_sentences_buffer = current_combined_chunk_content[-self.chunk_overlap_sentences:]
                else:
                    current_combined_chunk_content.append(sent)
                    current_combined_chunk_tokens += sent_tokens
                    current_overlap_sentences_buffer = current_combined_chunk_content[-self.chunk_overlap_sentences:]

        # Add the very last remaining chunk from this segment
        if current_combined_chunk_content:
            final_chunk_text = " ".join(current_combined_chunk_content)
            final_chunk_tokens = len(self.tokenizer.encode(final_chunk_text))

            if final_chunk_tokens >= self.min_chunk_size_tokens: # Check min size for last chunk too
                segment_chunks.append(DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    content=final_chunk_text,
                    chunk_index=chunk_idx_for_segment,
                    section_title=section_title,
                    subsection_title=subsection_title,
                    page_number=page_number
                ))
            else:
                logger.debug(f"Skipping final small chunk {chunk_idx_for_segment} from {doc_id}: {final_chunk_tokens} tokens.")

        return segment_chunks

    def _recursive_split_for_metadata_helper(self, text: str, current_separators: List[str], max_tokens: int, fallback_char_size: int) -> List[str]:
        """Helper for recursive splitting, extracted from the original _create_chunks logic."""
        if not text:
            return []

        text_tokens = len(self.tokenizer.encode(text))

        if text_tokens <= max_tokens:
            return [text]

        if not current_separators:
            sub_chunks = []
            for i in range(0, len(text), fallback_char_size):
                sub_chunk = text[i:i + fallback_char_size]
                sub_chunks.append(sub_chunk)
            return sub_chunks

        current_sep = current_separators[0]
        remaining_seps = current_separators[1:]

        split_parts = [part.strip() for part in text.split(current_sep) if part.strip()]

        collected_chunks_from_recursion = []
        current_group_of_parts = []
        current_group_tokens = 0

        for part in split_parts:
            part_tokens = len(self.tokenizer.encode(part))
            if current_group_tokens + part_tokens > max_tokens:
                if current_group_of_parts:
                    collected_chunks_from_recursion.extend(self._recursive_split_for_metadata_helper(
                        current_sep.join(current_group_of_parts), remaining_seps, max_tokens, fallback_char_size
                    ))
                collected_chunks_from_recursion.extend(self._recursive_split_for_metadata_helper(part, remaining_seps, max_tokens, fallback_char_size))
                current_group_of_parts = []
                current_group_tokens = 0
            else:
                current_group_of_parts.append(part)
                current_group_tokens += part_tokens

        if current_group_of_parts:
            collected_chunks_from_recursion.extend(self._recursive_split_for_metadata_helper(
                current_sep.join(current_group_of_parts), remaining_seps, max_tokens, fallback_char_size
            ))
        return collected_chunks_from_recursion

    def _extract_entities(self, content: str, doc_id: str) -> List[KnowledgeEntity]:
        """
        Extracts named entities using spaCy NER and custom PhraseMatcher rules.
        Includes a conceptual step for entity resolution/canonicalization, attempting to group variations like "Ground water" and "groundwater".
        """
        entities_found = []
        if not self.nlp:
            logger.error("SpaCy NLP model not loaded for entity extraction.")
            return entities_found

        doc = self.nlp(content)
        temp_entities_for_resolution = []

        # 1. Collect entities found by spaCy's default NER
        for ent in doc.ents:
            # Filter out very short or purely numeric entities that are unlikely to be meaningful names
            if len(ent.text.strip()) < 2 or ent.text.strip().strip('-.').isnumeric():
                continue
            # Store for later resolution
            temp_entities_for_resolution.append({
                'name': ent.text,
                'entity_type': ent.label_,
                'span': ent
            })

        # 2. Collect entities found by custom PhraseMatcher rules
        if self.phrase_matcher:
            matches = self.phrase_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                custom_entity_type = self.nlp.vocab.strings[match_id]

                if len(span.text.strip()) < 2 or span.text.strip().strip('-.').isnumeric():
                    continue

                # Prioritize custom types if they overlap with default NER, or add new ones
                temp_entities_for_resolution.append({
                    'name': span.text,
                    'entity_type': custom_entity_type,
                    'span': span
                })

        # 3. Conceptual Entity Resolution/Canonicalization
        resolved_entities_map = defaultdict(lambda: {'id': str(uuid.uuid4()), 'document_ids': [], 'related_chunks': [], 'mentions': []})

        for temp_ent in temp_entities_for_resolution:
            entity_name_raw = temp_ent['name']
            entity_type = temp_ent['entity_type']

            # Step A: Lowercase
            entity_name_lower = entity_name_raw.lower()

            # Step B: Remove common non-alphanumeric separators (hyphens, spaces) for canonical key generation
            entity_name_normalized = re.sub(r'[\s-]', '', entity_name_lower)

            # Use the normalized name + type as the canonical key
            canonical_key = f"{entity_name_normalized}::{entity_type}"

            # Update the canonical entity's information
            resolved_entities_map[canonical_key]['name'] = entity_name_raw
            resolved_entities_map[canonical_key]['entity_type'] = entity_type
            # Accumulate confidence (simple average or max could be used for more sophistication)
            resolved_entities_map[canonical_key]['confidence'] = max(resolved_entities_map[canonical_key].get('confidence', 0.0), temp_ent.get('confidence', 0.9)) # Take max confidence from mentions

            # Add document ID if not already present
            if doc_id not in resolved_entities_map[canonical_key]['document_ids']:
                resolved_entities_map[canonical_key]['document_ids'].append(doc_id)

            # Store the original mention for potential later use (e.g., for showing context)
            resolved_entities_map[canonical_key]['mentions'].append(entity_name_raw)

        # Convert resolved entities back to KnowledgeEntity objects
        entities = []
        for key, resolved_data in resolved_entities_map.items():
            entity = KnowledgeEntity(
                id=resolved_data['id'],
                name=resolved_data['name'],
                entity_type=resolved_data['entity_type'],
                confidence=min(resolved_data.get('confidence', 0.8), 1.0),
                document_ids=list(set(resolved_data['document_ids'])),
                related_chunks=[]
            )
            # IMPORTANT: KnowledgeEntity dataclass needs a 'mentions' field for this to work robustly in _create_relations
            if 'mentions' in resolved_data:
                entity.mentions = resolved_data['mentions']
            entities.append(entity)

        logger.info(f"Extracted {len(entities)} unique canonical entities from document {doc_id}.")
        return entities

    def _extract_keywords(self, chunks: List[DocumentChunk]):
        """Extract keywords using TF-IDF"""

        if not chunks:
            return

        try:
            texts = [chunk.content for chunk in chunks]
            non_empty_texts = [t for t in texts if t.strip()]

            if not non_empty_texts:
                logger.warning("No content in chunks for TF-IDF keyword extraction.")
                for chunk in chunks:
                    chunk.keywords = []
                return

            self.tfidf.fit(non_empty_texts)
            feature_names = self.tfidf.get_feature_names_out()

            for i, chunk in enumerate(chunks):
                if chunk.content.strip():
                    tfidf_matrix = self.tfidf.transform([chunk.content])
                    scores = tfidf_matrix.toarray()[0]
                    # Select top 5 keywords with score > 0
                    top_indices = scores.argsort()[-5:][::-1]
                    chunk.keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
                else:
                    chunk.keywords = []
        except Exception as e:
            logger.error(f"Error during TF-IDF keyword extraction: {e}")
            for chunk in chunks:
                chunk.keywords = []

    def _link_entities_to_chunks(self, chunks: List[DocumentChunk], entities: List[KnowledgeEntity]):
        """Link entities to chunks where they appear and add chunk IDs to entity.related_chunks."""

        entity_name_to_id = {ent.name.lower(): ent.id for ent in entities}

        for chunk in chunks:
            for ent_name_lower, ent_id in entity_name_to_id.items():
                if ent_name_lower in chunk.content.lower():
                    original_entity = next((e for e in entities if e.id == ent_id), None)
                    if original_entity:
                        if chunk.id not in original_entity.related_chunks:
                            original_entity.related_chunks.append(chunk.id)
                        if not any(e['id'] == ent_id for e in chunk.entities):
                            chunk.entities.append({
                                'id': original_entity.id,
                                'name': original_entity.name,
                                'type': original_entity.entity_type
                            })

    def _create_relations(self, entities: List[KnowledgeEntity], full_content: str, doc_id: str, chunks: List[DocumentChunk]) -> Tuple[List[KnowledgeRelation], List[KnowledgeEntity]]:
        """ Create relations using spaCy's dependency parser and rule-based patterns. Falls back to co-occurrence if patterns don't match. """

        relations = []
        if not self.nlp or not self.matcher:
            return relations, entities

        doc = self.nlp(full_content)
        # Use a more robust mapping from any mention to its canonical entity ID
        mention_to_entity_id = {}
        for entity_obj in entities:
            # Assumes KnowledgeEntity has a 'mentions' field
            for mention in getattr(entity_obj, 'mentions', [entity_obj.name]):
                mention_to_entity_id[mention.lower()] = entity_obj.id

        # Rule-based extraction using spaCy Matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            rule_name = self.nlp.vocab.strings[match_id]

            span_entities_in_kg = []
            for ent_in_span in span.ents:
                if ent_in_span.text.lower() in mention_to_entity_id:
                    canonical_entity_id = mention_to_entity_id[ent_in_span.text.lower()]
                    canonical_entity_obj = next((e for e in entities if e.id == canonical_entity_id), None)
                    if canonical_entity_obj and canonical_entity_obj not in span_entities_in_kg:
                        span_entities_in_kg.append(canonical_entity_obj)

            # --- MODIFIED: Handle new methodological relation types FIRST ---
            if rule_name == "ENSURES_INDEPENDENCE":
                ri_entity = next((e for e in entities if e.name.lower() == "rapid intensification" or e.entity_type == "HYDRO_EVENT"), None)
                if ri_entity:
                    relations.append(KnowledgeRelation(
                        id=str(uuid.uuid4()),
                        source_entity=ri_entity.id,
                        target_entity=span.text, # The textual mention of MST/declustering as the target
                        relation_type="INDEPENDENCE_ENSURED_BY",
                        confidence=0.95,
                        document_id=doc_id,
                        sentence=span.text
                    ))
                continue # Move to the next match

            elif rule_name == "DEFINES_THRESHOLD":
                mhw_entity = next((e for e in entities if e.name.lower() == "marine heatwave" or e.entity_type == "MARINE_HEATWAVE"), None)
                if mhw_entity:
                    relations.append(KnowledgeRelation(
                        id=str(uuid.uuid4()),
                        source_entity=mhw_entity.id,
                        target_entity=span.text, # The definition text as the target
                        relation_type="DEFINED_BY_THRESHOLD",
                        confidence=0.9,
                        document_id=doc_id,
                        sentence=span.text
                    ))
                continue # Move to the next match

            elif rule_name == "QUANTIFIES_IMPACT":
                mhw_entity = next((e for e in entities if e.name.lower() == "marine heatwave" or e.entity_type == "MARINE_HEATWAVE"), None)
                ri_entity = next((e for e in entities if e.name.lower() == "rapid intensification" or e.entity_type == "HYDRO_EVENT"), None)
                if mhw_entity and ri_entity:
                    relations.append(KnowledgeRelation(
                        id=str(uuid.uuid4()),
                        source_entity=span.text, # "Multiplication Rate" as the source concept
                        target_entity=f"RI_MHW_Interaction_{doc_id}", # A synthetic node for the interaction or link directly to both
                        relation_type="QUANTIFIES_IMPACT_ON",
                        confidence=0.95,
                        document_id=doc_id,
                        sentence=span.text
                    ))
                continue # Move to the next match

            elif rule_name == "USES_DATASET":
                # Ensure a study entity exists for this document to link the dataset to
                study_entity = next((e for e in entities if e.entity_type == "STUDY" and doc_id in e.document_ids), None)
                if not study_entity: # If no explicit "study" entity for this doc, create a generic one
                    study_entity = KnowledgeEntity(id=f"study_concept_{doc_id}", name=f"Analysis of {doc_id}", entity_type="STUDY", confidence=1.0, document_ids=[doc_id])
                    entities.append(study_entity)

                dataset_names_in_span = [token.text for token in span if token.text.upper() in ["IBTRACS", "ERA5", "NOAA", "OISST", "COPENICUS"]]
                for dataset_name in dataset_names_in_span:
                    # Try to find existing dataset entity first, matching by name or mention
                    dataset_entity = next((e for e in entities if e.name == dataset_name or e.name.lower() == dataset_name.lower() or dataset_name.upper() in [m.upper() for m in getattr(e, 'mentions', [])] ), None)
                    if not dataset_entity: # Create if not already an entity
                        dataset_entity = KnowledgeEntity(id=f"dataset_{dataset_name.lower()}_{doc_id}", name=dataset_name, entity_type="DATASET", confidence=0.8, document_ids=[doc_id])
                        entities.append(dataset_entity) # NEW: Add newly created entity to the list

                    relations.append(KnowledgeRelation(
                        id=str(uuid.uuid4()),
                        source_entity=study_entity.id,
                        target_entity=dataset_entity.id,
                        relation_type="USES_DATASET",
                        confidence=0.95,
                        document_id=doc_id,
                        sentence=span.text
                    ))
                continue

            if len(span_entities_in_kg) >= 2:
                source_ent = span_entities_in_kg[0]
                target_ent = span_entities_in_kg[1]

                if source_ent.id != target_ent.id:

                    relation_type = rule_name

                    # Refine relation_type based on specific entity types involved
                    if rule_name == "FLOWS_INTO" and (source_ent.entity_type in ["WATER_BODY", "GPE"]) and \
                        (target_ent.entity_type in ["WATER_BODY", "GPE"]):
                        relation_type = "FLOWS_INTO"
                    elif rule_name == "MEASURES" and source_ent.entity_type == "HYDRO_INFRASTRUCTURE" and \
                        target_ent.entity_type in ["HYDRO_MEASUREMENT", "POLLUTANT"]:
                        relation_type = "MEASURES"
                    elif rule_name == "CONTAINS" and source_ent.entity_type == "WATER_BODY" and \
                        target_ent.entity_type == "POLLUTANT":
                        relation_type = "CONTAINS"
                    elif rule_name == "CAUSES" and source_ent.entity_type == "HYDRO_EVENT" and \
                        target_ent.entity_type in ["HYDRO_EVENT", "HYDRO_MEASUREMENT"]:
                        relation_type = "CAUSES"
                    elif rule_name == "USES_MODEL" and source_ent.entity_type in ["ORG", "PERSON"] and \
                        target_ent.entity_type == "HYDRO_MODEL":
                        relation_type = "USES_MODEL"
                    elif rule_name == "INVESTIGATES" and source_ent.entity_type == "STUDY" and \
                        target_ent.entity_type in ["WATER_BODY", "HYDRO_EVENT", "POLLUTANT", "HYDRO_MEASUREMENT"]:
                        relation_type = "INVESTIGATES"
                    # Else, use the rule_name as the relation_type (e.g., "RELATED_TO_VIA_RULE")

                    relations.append(KnowledgeRelation(
                        id=str(uuid.uuid4()),
                        source_entity=source_ent.id,
                        target_entity=target_ent.id,
                        relation_type=relation_type,
                        confidence=0.9,
                        document_id=doc_id,
                        sentence=span.text
                    ))

        # Fallback to co-occurrence for entities in the same sentence if no specific rule applied
        extracted_relations_set = set([(r.source_entity, r.target_entity, r.relation_type) for r in relations])

        for sent in doc.sents:
            sent_entities = []
            # Gather entities from the sentence that exist in our main `entity_map`
            for ent_in_sent in sent.ents:
                if ent_in_sent.text.lower() in mention_to_entity_id:
                    canonical_entity_id = mention_to_entity_id[ent_in_sent.text.lower()]
                    canonical_entity_obj = next((e for e in entities if e.id == canonical_entity_id), None)
                    if canonical_entity_obj and canonical_entity_obj not in sent_entities:
                        sent_entities.append(canonical_entity_obj)

            if self.phrase_matcher:
                phrase_matches_in_sent = self.phrase_matcher(sent)
                for match_id, start, end in phrase_matches_in_sent:
                    span_in_sent = sent.char_span(start, end)
                    if span_in_sent and span_in_sent.text.lower() in mention_to_entity_id:
                        canonical_entity_id = mention_to_entity_id[span_in_sent.text.lower()]
                        canonical_entity_obj = next((e for e in entities if e.id == canonical_entity_id), None)
                        if canonical_entity_obj and canonical_entity_obj not in sent_entities:
                            sent_entities.append(canonical_entity_obj)

            if len(sent_entities) >= 2:
                for i, ent1 in enumerate(sent_entities):
                    for j, ent2 in enumerate(sent_entities):
                        if i < j:
                            specific_relation_exists = False
                            for r_check in extracted_relations_set:
                                # Check both (ent1, ent2) and (ent2, ent1) directions for any existing relation type
                                if (r_check[0] == ent1.id and r_check[1] == ent2.id) or (r_check[0] == ent2.id and r_check[1] == ent1.id):
                                    specific_relation_exists = True
                                    break

                            if not specific_relation_exists:
                                relation = KnowledgeRelation(
                                    id=str(uuid.uuid4()),
                                    source_entity=ent1.id,
                                    target_entity=ent2.id,
                                    relation_type='CO_OCCURS',
                                    confidence=0.5,
                                    document_id=doc_id,
                                    sentence=sent.text
                                )
                                relations.append(relation)
                                extracted_relations_set.add((relation.source_entity, relation.target_entity, relation.relation_type))
        return relations, entities

# 4. VECTOR INDEX MANAGER
# =============================================================================

class VectorIndexManager:
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: Dict[str, Any] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.current_index = 0

        self.nlist = 15 # Number of inverted lists (clusters). [future: 100]
        self.nprobe = 5 # Number of lists to search at query time. Higher = more accurate, slower. [future: 20]
        self.training_vectors_limit = 10000 # Max number of vectors to use for training the index.

        self._initialize_index()

    def _initialize_index(self):
        """Initializes a new FAISS IndexIVFFlat index."""

        # The quantizer is a simple flat index that the IVF index uses to cluster vectors
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        # Create the IndexIVFFlat index
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.make_direct_map() # Enables direct mapping from internal FAISS IDs to external IDs
        logger.info(f"Initialized new FAISS IndexIVFFlat with nlist={self.nlist}.")

    def add_chunks(self, chunks: List['DocumentChunk']): # Use forward reference for DocumentChunk
        """Add chunks to vector index, performing training if necessary."""

        if not chunks:
            return

        embeddings_to_add = []
        new_chunk_ids = []
        for chunk in chunks:
            if chunk.embedding is not None and chunk.id not in self.id_to_index: # Only add new chunks
                embeddings_to_add.append(chunk.embedding)
                new_chunk_ids.append(chunk.id)
                # Store metadata and mappings for new chunks
                self.chunk_metadata[chunk.id] = chunk.to_dict()
                self.id_to_index[chunk.id] = self.current_index
                self.index_to_id[self.current_index] = chunk.id
                self.current_index += 1
            elif chunk.id in self.id_to_index:
                # Update existing chunk metadata if content might have changed
                self.chunk_metadata[chunk.id] = chunk.to_dict()

        if embeddings_to_add:
            embeddings_array = np.array(embeddings_to_add).astype('float32')

            # IndexIVFFlat requires training before adding vectors.
            if not self.index.is_trained:
                logger.info(f"FAISS index is not trained. Training with {min(len(embeddings_array), self.training_vectors_limit)} vectors.")
                # Use a subset of vectors for training if the dataset is very large
                if len(embeddings_array) > self.training_vectors_limit:
                    # Randomly sample vectors for training
                    training_indices = np.random.choice(len(embeddings_array), self.training_vectors_limit, replace=False)
                    training_data = embeddings_array[training_indices]
                else:
                    training_data = embeddings_array

                self.index.train(training_data)
                logger.info("FAISS Index training complete.")

            self.index.add(embeddings_array)
            logger.info(f"Added {len(embeddings_to_add)} new embeddings to FAISS index. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using the IVF index."""

        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. No search results.")
            return []

        if not self.index.is_trained:
            logger.error("FAISS index is not trained. Cannot perform search. Please add documents first.")
            return []

        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.index_to_id:
                chunk_id = self.index_to_id[idx]
                chunk_dict = self.chunk_metadata.get(chunk_id, {})

                if chunk_dict:
                    result_item = {
                        'chunk_id': chunk_id,
                        'similarity_score': float(dist),
                        'content': chunk_dict.get('content', ''),
                        'document_id': chunk_dict.get('document_id', ''),
                        'keywords': chunk_dict.get('keywords', []),
                        'entities': chunk_dict.get('entities', [])
                    }
                    results.append(result_item)
                else:
                    logger.warning(f"Index {idx} not found in chunk_metadata for chunk_id {chunk_id}. Data inconsistency.")
            else:
                logger.warning(f"Index {idx} not found in index_to_id mapping. Data inconsistency.")
        return results

    def save_index(self):
        """Saves the FAISS index and associated metadata."""

        try:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            with open(CHUNK_METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump({cid: data for cid, data in self.chunk_metadata.items()}, f, indent=2)
            with open(ID_TO_INDEX_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.id_to_index, f, indent=2)
            with open(INDEX_TO_ID_PATH, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.index_to_id.items()}, f, indent=2)
            logger.info("FAISS index and metadata saved.")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False

    def load_index(self):
        """Loads the FAISS index and associated metadata."""

        if (FAISS_INDEX_PATH.exists() and CHUNK_METADATA_PATH.exists() and
            ID_TO_INDEX_PATH.exists() and INDEX_TO_ID_PATH.exists()):
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(CHUNK_METADATA_PATH, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                with open(ID_TO_INDEX_PATH, 'r', encoding='utf-8') as f:
                    self.id_to_index = json.load(f)
                with open(INDEX_TO_ID_PATH, 'r', encoding='utf-8') as f:
                    self.index_to_id = {int(k): v for k, v in json.load(f).items()}
                self.current_index = self.index.ntotal
                logger.info(f"FAISS index and metadata loaded. Total items: {self.index.ntotal}")
                return True
            except Exception as e:
                logger.error(f"Error loading FAISS index, restarting from scratch: {e}")
                self._initialize_index()
                return False
        logger.info("No existing FAISS index found. Starting fresh.")
        return False

# 5. KNOWLEDGE GRAPH MANAGER
# =============================================================================

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.document_chunks: Dict[str, DocumentChunk] = {}

    def add_data(self, chunks: List[DocumentChunk], entities: List[KnowledgeEntity], relations: List[KnowledgeRelation]):
        """Add entities, relations, and chunks to the graph."""

        for chunk in chunks:
            if chunk.id not in self.document_chunks:
                self.document_chunks[chunk.id] = chunk

            self.graph.add_node(chunk.id, type='CHUNK', content=chunk.content[:200] + '...', document_id=chunk.document_id, chunk_index=chunk.chunk_index,
                                section_title=chunk.section_title, subsection_title=chunk.subsection_title, page_number=chunk.page_number)

        for entity in entities:
            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, name=entity.name, type=entity.entity_type, document_ids=entity.document_ids, related_chunks=entity.related_chunks, mentions=entity.mentions)
            else:
                existing_entity = self.entities[entity.id]
                existing_entity.document_ids.extend([d_id for d_id in entity.document_ids if d_id not in existing_entity.document_ids])
                existing_entity.related_chunks.extend([c_id for c_id in entity.related_chunks if c_id not in existing_entity.related_chunks])
                existing_entity.mentions.extend([m for m in entity.mentions if m not in existing_entity.mentions])
                self.graph.nodes[entity.id]['name'] = existing_entity.name
                self.graph.nodes[entity.id]['type'] = existing_entity.entity_type
                self.graph.nodes[entity.id]['document_ids'] = existing_entity.document_ids
                self.graph.nodes[entity.id]['related_chunks'] = existing_entity.related_chunks
                self.graph.nodes[entity.id]['mentions'] = existing_entity.mentions

        for entity in entities:
            for chunk_id in entity.related_chunks:
                if chunk_id in self.graph and entity.id in self.graph:
                    if self.graph.get_edge_data(chunk_id, entity.id, key='MENTIONS') is None:
                        self.graph.add_edge(chunk_id, entity.id, key='MENTIONS', relation_type='MENTIONS', confidence=1.0)
                    if self.graph.get_edge_data(entity.id, chunk_id, key='MENTIONED_IN') is None:
                        self.graph.add_edge(entity.id, chunk_id, key='MENTIONED_IN', relation_type='MENTIONED_IN', confidence=1.0)
                else:
                    logger.debug(f"Skipping MENTIONS relation: Chunk {chunk_id} or Entity {entity.id} not found in graph for direct linking.")

        # Add relations (between entities)
        added_relation_tuples = set()

        for relation in relations:
            # Ensure both source and target nodes exist in the graph before adding edge
            if relation.source_entity in self.graph and relation.target_entity in self.graph:
                relation_tuple = (relation.source_entity, relation.target_entity, relation.relation_type)
                if relation_tuple not in added_relation_tuples:
                    if self.graph.get_edge_data(relation.source_entity, relation.target_entity, key=relation.id) is None:
                        self.relations[relation.id] = relation
                        self.graph.add_edge(
                            relation.source_entity,
                            relation.target_entity,
                            key=relation.id,
                            relation_type=relation.relation_type,
                            confidence=relation.confidence,
                            document_id=relation.document_id,
                            sentence=relation.sentence
                        )
                        added_relation_tuples.add(relation_tuple)
                else:
                    logger.debug(f"Skipping semantically duplicate relation type '{relation.relation_type}' between {relation.source_entity} and {relation.target_entity}.")
            else:
                logger.warning(f"Skipping relation {relation.id}: source ({relation.source_entity}) or target ({relation.target_entity}) entity not in graph during relation addition.")

        logger.info(f"KG Population Status: Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}.")
        logger.info(f"Total entities stored: {len(self.entities)}, Total relations stored: {len(self.relations)}, Total document chunks stored: {len(self.document_chunks)}")

    def get_related_info(self, query_entities: List[str], max_distance: int = 2) -> List[Dict[str, Any]]:
        """ Get entities and related chunks connected to the given query entities, traversing the knowledge graph. """
        related_info = []
        seen_nodes = set()

        query_entity_ids = []
        for q_ent_name in query_entities:
            for eid, entity_obj in self.entities.items():
                if entity_obj.name.lower() == q_ent_name.lower() or q_ent_name.lower() in [m.lower() for m in entity_obj.mentions]:
                    query_entity_ids.append(eid)
                    break
        if not query_entity_ids:
            return []

        for start_node_id in query_entity_ids:
            if start_node_id not in self.graph:
                continue

            # BFS to find nodes within max_distance
            for target_node in nx.bfs_tree(self.graph, start_node_id, depth_limit=max_distance).nodes():
                if target_node == start_node_id:
                    continue

                if target_node not in seen_nodes:
                    node_data = self.graph.nodes[target_node]
                    try:
                        distance = nx.shortest_path_length(self.graph, source=start_node_id, target=target_node)
                    except nx.NetworkXNoPath:
                        continue

                    info_item = {
                        'id': target_node,
                        'type': node_data.get('type'),
                        'distance_from_query_entity': distance
                    }

                    if node_data.get('type') == 'CHUNK':
                        chunk_obj = self.document_chunks.get(target_node)
                        if chunk_obj:
                            info_item['content_snippet'] = chunk_obj.content[:200] + "..."
                            info_item['document_id'] = chunk_obj.document_id
                            info_item['page_number'] = chunk_obj.page_number
                            info_item['section_title'] = chunk_obj.section_title
                        else:
                            info_item['content_snippet'] = node_data.get('content', '')
                            info_item['document_id'] = "UNKNOWN"
                    elif node_data.get('type') in ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC', 'LOCATION', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'STUDY', 'WATER_BODY', 'HYDRO_MEASUREMENT', 'POLLUTANT', 'HYDRO_EVENT', 'HYDRO_INFRASTRUCTURE', 'HYDRO_MODEL', 'DATASET']: # Added DATASET
                        info_item['name'] = node_data.get('name')
                        if self.graph.has_edge(start_node_id, target_node):
                            connecting_edge_data = self.graph.get_edge_data(start_node_id, target_node)
                            if connecting_edge_data:
                                found_edge_attrs = None
                                for key in connecting_edge_data:
                                    edge_attrs = connecting_edge_data[key]
                                    if edge_attrs.get('relation_type') not in ['MENTIONS', 'MENTIONED_IN']:
                                        found_edge_attrs = edge_attrs
                                        break
                                if not found_edge_attrs and connecting_edge_data:
                                    found_edge_attrs = list(connecting_edge_data.values())[0]

                                if found_edge_attrs:
                                    info_item['connection_type'] = found_edge_attrs.get('relation_type', 'unknown')
                                    info_item['connection_confidence'] = found_edge_attrs.get('confidence', 0.0)
                    related_info.append(info_item)
                    seen_nodes.add(target_node)
        return related_info

    def add_data(self, chunks: List[DocumentChunk], entities: List[KnowledgeEntity], relations: List[KnowledgeRelation]):
        """Add entities, relations, and chunks to the graph."""

        for chunk in chunks:
            if chunk.id not in self.document_chunks:
                self.document_chunks[chunk.id] = chunk

            # Ensure attributes are GML-compatible (strings, not None)
            chunk_attrs = {
                'type': 'CHUNK',
                'content': chunk.content[:200] + '...',
                'document_id': chunk.document_id,
                'chunk_index': str(chunk.chunk_index), # Convert int to string for GML
                'section_title': chunk.section_title if chunk.section_title is not None else "N/A", # Handle None
                'subsection_title': chunk.subsection_title if chunk.subsection_title is not None else "N/A", # Handle None
                'page_number': str(chunk.page_number) if chunk.page_number is not None else "N/A" # Convert int to string, handle None
            }
            self.graph.add_node(chunk.id, **chunk_attrs)

        for entity in entities:
            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                # For entities, ensure attributes are strings/lists of strings
                entity_attrs = {
                    'name': entity.name,
                    'type': entity.entity_type,
                    'document_ids': json.dumps(entity.document_ids), # Convert list to JSON string
                    'related_chunks': json.dumps(entity.related_chunks), # Convert list to JSON string
                    'mentions': json.dumps(entity.mentions) # Convert list to JSON string
                }
                self.graph.add_node(entity.id, **entity_attrs)
            else:
                existing_entity = self.entities[entity.id]
                # Update logic (ensure updates also handle None/type consistency if saving the whole graph again)
                existing_entity.document_ids.extend([d_id for d_id in entity.document_ids if d_id not in existing_entity.document_ids])
                existing_entity.related_chunks.extend([c_id for c_id in entity.related_chunks if c_id not in existing_entity.related_chunks])
                existing_entity.mentions.extend([m for m in entity.mentions if m not in existing_entity.mentions])

                # Update graph node attributes explicitly for existing nodes
                if entity.id in self.graph:
                    self.graph.nodes[entity.id]['name'] = existing_entity.name
                    self.graph.nodes[entity.id]['type'] = existing_entity.entity_type
                    self.graph.nodes[entity.id]['document_ids'] = json.dumps(existing_entity.document_ids)
                    self.graph.nodes[entity.id]['related_chunks'] = json.dumps(existing_entity.related_chunks)
                    self.graph.nodes[entity.id]['mentions'] = json.dumps(existing_entity.mentions)

        for entity in entities:
            for chunk_id in entity.related_chunks:
                if chunk_id in self.graph and entity.id in self.graph:
                    # MENTIONS/MENTIONED_IN edges
                    # These have simple string attributes, should be fine
                    if self.graph.get_edge_data(chunk_id, entity.id, key='MENTIONS') is None:
                        self.graph.add_edge(chunk_id, entity.id, key='MENTIONS', relation_type='MENTIONS', confidence=1.0)
                    if self.graph.get_edge_data(entity.id, chunk_id, key='MENTIONED_IN') is None:
                        self.graph.add_edge(entity.id, chunk_id, key='MENTIONED_IN', relation_type='MENTIONED_IN', confidence=1.0)
                else:
                    logger.debug(f"Skipping MENTIONS relation: Chunk {chunk_id} or Entity {entity.id} not found in graph for direct linking.")

        # Add relations (between entities)
        added_relation_tuples = set()

        for relation in relations:
            # Ensure both source and target nodes exist in the graph before adding edge
            if relation.source_entity in self.graph and relation.target_entity in self.graph:
                relation_tuple = (relation.source_entity, relation.target_entity, relation.relation_type)
                if relation_tuple not in added_relation_tuples:
                    if self.graph.get_edge_data(relation.source_entity, relation.target_entity, key=relation.id) is None:
                        self.relations[relation.id] = relation
                        # Ensure relation attributes are GML-compatible (strings, not None)
                        edge_attrs = {
                            'relation_type': relation.relation_type,
                            'confidence': float(relation.confidence), # GML might prefer float over numpy types or specific precision
                            'document_id': relation.document_id if relation.document_id is not None else "N/A", # Handle None
                            'sentence': relation.sentence if relation.sentence is not None else "N/A" # Handle None
                        }
                        self.graph.add_edge(
                            relation.source_entity,
                            relation.target_entity,
                            key=relation.id,
                            **edge_attrs
                        )
                        added_relation_tuples.add(relation_tuple)
            else:
                logger.warning(f"Skipping relation {relation.id}: source ({relation.source_entity}) or target ({relation.target_entity}) entity not in graph during relation addition.")

    def save_graph(self) -> bool:
        """Saves the NetworkX graph and associated entity/relation data."""
        try:
            # KG_GRAPH_PATH is a Path object, so convert to string
            nx.write_gml(self.graph, str(KG_GRAPH_PATH))

            with open(KG_ENTITIES_PATH, 'w', encoding='utf-8') as f:
                json.dump({eid: entity.to_dict() for eid, entity in self.entities.items()}, f, indent=2)
            with open(KG_RELATIONS_PATH, 'w', encoding='utf-8') as f:
                json.dump({rid: relation.to_dict() for rid, relation in self.relations.items()}, f, indent=2)

            # Save document chunks (metadata only, no embeddings)
            chunks_to_save = {cid: chunk.to_dict() for cid, chunk in self.document_chunks.items()}
            for cid in chunks_to_save:
                # Ensure embedding is None or removed before JSON serialization
                if 'embedding' in chunks_to_save[cid]:
                    chunks_to_save[cid]['embedding'] = None
            with open(KG_CHUNKS_PATH, 'w', encoding='utf-8') as f:
                json.dump(chunks_to_save, f, indent=2)

            logger.info("Knowledge Graph and associated data saved.")
            return True
        except Exception as e:
            logger.error(f"Error saving Knowledge Graph: {e}")
            return False

    def load_graph(self) -> bool:
        """Loads the NetworkX graph and associated entity/relation data."""

        if (KG_GRAPH_PATH.exists() and KG_ENTITIES_PATH.exists() and
            KG_RELATIONS_PATH.exists() and KG_CHUNKS_PATH.exists()):
            try:
                self.graph = nx.read_gml(str(KG_GRAPH_PATH))

                with open(KG_ENTITIES_PATH, 'r', encoding='utf-8') as f:
                    self.entities = {eid: KnowledgeEntity.from_dict(data) for eid, data in json.load(f).items()}
                with open(KG_RELATIONS_PATH, 'r', encoding='utf-8') as f:
                    self.relations = {rid: KnowledgeRelation.from_dict(data) for rid, data in json.load(f).items()}
                with open(KG_CHUNKS_PATH, 'r', encoding='utf-8') as f:
                    self.document_chunks = {cid: DocumentChunk.from_dict(data) for cid, data in json.load(f).items()}

                logger.info(f"Knowledge Graph loaded. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
                return True
            except Exception as e:
                logger.error(f"Error loading Knowledge Graph, restarting from scratch: {e}")
                self.graph = nx.MultiDiGraph()
                return False
        logger.info("No existing Knowledge Graph found. Starting fresh.")
        return False

# 6. QUERY PROCESSOR
# =============================================================================

class QueryProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            logger.error(f"spaCy model '{SPACY_MODEL}' not found for QueryProcessor. Install with: python -m spacy download {SPACY_MODEL}")
            self.nlp = None

        self.query_types = {
            'factual': ['what', 'who', 'when', 'where', 'which', 'how many'],
            'explanatory': ['how', 'why', 'explain', 'describe', 'tell me about'],
            'comparative': ['compare', 'versus', 'difference', 'similarities'],
            'procedural': ['steps', 'process', 'procedure', 'how to'],
            'definition': ['what is', 'define']
        }
        self.stop_words = self.nlp.Defaults.stop_words if self.nlp else set()

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and extract entities/keywords, including decomposition."""

        query_lower = query.lower()
        doc = self.nlp(query) if self.nlp else None

        # Classify query type
        query_type = 'general'
        for qtype, indicators in self.query_types.items():
            if any(indicator in query_lower for indicator in indicators):
                query_type = qtype
                break

        # Extract keywords (more robust with spaCy)
        keywords = []
        if doc:
            keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and not token.is_punct and len(token.lemma_) > 2]
            keywords = list(set(keywords))

        # Extract entities using spaCy's NER
        entities = []
        if doc:
            for ent in doc.ents:
                entities.append({'text': ent.text, 'label': ent.label_})

        relevant_entity_labels = ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC', 'STUDY', 'WATER_BODY', 'HYDRO_MEASUREMENT', 'POLLUTANT', 'HYDRO_EVENT', 'HYDRO_INFRASTRUCTURE', 'HYDRO_MODEL', 'DATASET', 'MARINE_HEATWAVE', 'RAPID_INTENSIFICATION'] # Added example custom types
        extracted_entity_names = [e['text'] for e in entities if e['label'] in relevant_entity_labels]

        sub_queries = []
        if "independent ri events" in query_lower or "how ensured ri independence" in query_lower or "how independent" in query_lower:
            sub_queries.append("How independence of rapid intensification events was ensured")
        if "thresholds used for identifying influential mhws" in query_lower or "mhw thresholds" in query_lower or "specific thresholds" in query_lower:
            sub_queries.append("Specific thresholds for identifying influential marine heatwaves")
        if "multiplication rate quantifies" in query_lower or "how multiplication rate measures impact" in query_lower or "quantifies the mhw impact" in query_lower:
            sub_queries.append("How the multiplication rate quantifies marine heatwave impact")
        if "data sources" in query_lower or "what data was used" in query_lower or "data sets" in query_lower:
            sub_queries.append("Critical data sources for probabilistic analysis")

        # Extract specific methodological terms for targeted retrieval
        methodological_terms = []
        if doc:
            for token in doc:
                lemma_lower = token.lemma_.lower()
                if lemma_lower in ["mst", "minimum separation time", "declustering scheme", "double-threshold", "multiplication rate"]:
                    methodological_terms.append(token.text)
                if lemma_lower in ["ibtracs", "era5", "oisst", "noaa", "copernicus"]: # Dataset names as terms
                    methodological_terms.append(token.text)

        methodological_terms = list(set(methodological_terms))
        sub_queries = list(set(sub_queries))

        return {
            'original_query': query,
            'query_type': query_type,
            'keywords': keywords,
            'entities': extracted_entity_names,
            'complexity': 'simple' if len(query.split()) < 5 else 'medium' if len(query.split()) < 15 else 'complex',
            'sub_queries': sub_queries, # NEW
            'methodological_terms': methodological_terms # NEW
        }

    def _resolve_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Placeholder for entity resolution/linking. This would typically involve calling an external service or a more complex model to link entities to canonical IDs (e.g., Wikidata, DBpedia). """

        resolved = []
        for ent in entities:
            resolved_ent = ent.copy()
            resolved_ent['canonical_id'] = None
            resolved.append(resolved_ent)
        return resolved


# 7. HYBRID RETRIEVAL ENGINE
# =============================================================================

class HybridRetrievalEngine:
    def __init__(self, vector_index: VectorIndexManager, kg_manager: KnowledgeGraphManager, embedding_service: EmbeddingService):
        self.vector_index = vector_index
        self.kg_manager = kg_manager
        self.query_processor = QueryProcessor()
        self.embedding_service = embedding_service

        self.cross_encoder = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining vector search and knowledge graph, with re-ranking."""

        query_analysis = self.query_processor.analyze_query(query)

        all_candidates = []
        seen_chunk_ids = set()

        # 1. Retrieve based on the original query
        original_query_embedding = self.embedding_service.get_embedding(query)
        vector_candidates_main = self.vector_index.search(original_query_embedding, top_k * 5)
        for candidate in vector_candidates_main:
            if candidate['chunk_id'] not in seen_chunk_ids:
                all_candidates.append(candidate)
                seen_chunk_ids.add(candidate['chunk_id'])

        # 2. Retrieve for explicit methodological terms
        for term in query_analysis.get('methodological_terms', []):
            term_embedding = self.embedding_service.get_embedding(term)
            vector_candidates_term = self.vector_index.search(term_embedding, top_k * 2) # Retrieve fewer for terms as they are very specific
            for candidate in vector_candidates_term:
                if candidate['chunk_id'] not in seen_chunk_ids:
                    all_candidates.append(candidate)
                    seen_chunk_ids.add(candidate['chunk_id'])

        # 3. Retrieve for decomposed sub-queries
        for decomposed_query_text in query_analysis.get('sub_queries', []):
            decomposed_query_embedding = self.embedding_service.get_embedding(decomposed_query_text)
            vector_candidates_sub = self.vector_index.search(decomposed_query_embedding, top_k * 3) # Moderate number of candidates
            for candidate in vector_candidates_sub:
                if candidate['chunk_id'] not in seen_chunk_ids:
                    all_candidates.append(candidate)
                    seen_chunk_ids.add(candidate['chunk_id'])

        # 4. Knowledge Graph Enhancement
        # Use a broader set of entities/terms for KG traversal
        kg_query_entities = query_analysis.get('entities', []) + query_analysis.get('methodological_terms', [])
        kg_related_info = self.kg_manager.get_related_info(kg_query_entities)

        # Boost all gathered candidates
        enhanced_candidates = self._enhance_and_boost_candidates(all_candidates, query_analysis, kg_related_info)

        # Add KG-derived chunks as candidates if they are not already in enhanced_candidates
        kg_chunk_candidates = []
        # Use the consolidated `seen_chunk_ids` from multi-stage retrieval

        for item in kg_related_info:
            if item['type'] == 'CHUNK':
                chunk_id = item['id']
                if chunk_id not in seen_chunk_ids: # Ensure it's not already added from vector search
                    chunk_data = self.vector_index.chunk_metadata.get(chunk_id)
                    if chunk_data:
                        kg_score = max(0.1, 1.0 - (item['distance_from_query_entity'] * 0.2))
                        candidate_chunk = {
                            'chunk_id': chunk_id,
                            'similarity_score': kg_score,
                            'content': chunk_data.get('content', ''),
                            'document_id': chunk_data.get('document_id', ''),
                            'keywords': chunk_data.get('keywords', []),
                            'entities': chunk_data.get('entities', []),
                            'section_title': chunk_data.get('section_title'), # NEW
                            'subsection_title': chunk_data.get('subsection_title'), # NEW
                            'page_number': chunk_data.get('page_number') # NEW
                        }
                        enhanced_candidates.append(candidate_chunk)
                        seen_chunk_ids.add(chunk_id)

        # 5. Apply Re-ranking (Cross-encoder or RRF)
        final_results = self._rank_results(enhanced_candidates, query_analysis, query, top_k)

        return final_results[:top_k]

    def _enhance_and_boost_candidates(self, candidates: List[Dict[str, Any]], query_analysis: Dict[str, Any], kg_related_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Enhance and boost scores of candidates based on KG relatedness, keyword overlap, section relevance, and specific terms. """

        enhanced_results = []
        query_entities = [e.lower() for e in query_analysis.get('entities', [])]
        query_keywords = [kw.lower() for kw in query_analysis.get('keywords', [])]
        query_methodological_terms = [t.lower() for t in query_analysis.get('methodological_terms', [])]
        query_sub_queries = [sq.lower() for sq in query_analysis.get('sub_queries', [])]


        for result in candidates:
            enhanced_result = result.copy()
            initial_score = enhanced_result.get('similarity_score', 0)

            # --- Existing boosting for KG entity overlap and direct keyword matches ---
            doc_entities = [e.get('name', '').lower() for e in result.get('entities', [])]
            entity_overlap_count = len(set(query_entities) & set(doc_entities))
            if entity_overlap_count > 0:
                initial_score *= (1 + entity_overlap_count * 0.2)

            content_lower = result.get('content', '').lower()
            doc_keywords = [kw.lower() for kw in result.get('keywords', [])]
            keyword_content_matches = sum(1 for kw in query_keywords if kw in content_lower)
            keyword_tag_matches = len(set(query_keywords) & set(doc_keywords))
            total_keyword_matches = keyword_content_matches + keyword_tag_matches
            if total_keyword_matches > 0:
                initial_score *= (1 + total_keyword_matches * 0.05)

            is_kg_related = False
            for kg_item in kg_related_info:
                if kg_item['type'] == 'CHUNK' and kg_item['id'] == result['chunk_id']:
                    is_kg_related = True
                    break
                if kg_item['type'] != 'CHUNK' and any(e['id'] == kg_item['id'] for e in result.get('entities', [])):
                    is_kg_related = True
                    break
            if is_kg_related:
                initial_score *= 1.1

            section_lower = result.get('section_title', '').lower()
            subsection_lower = result.get('subsection_title', '').lower()

            # Broad section relevance
            if "method" in section_lower or "discussion" in section_lower or "results" in section_lower:
                initial_score *= 1.05

            # Stronger boost if query explicitly asks for data sources and chunk is from data section
            if "critical data sources for probabilistic analysis" in query_sub_queries:
                if "data" in section_lower or "data" in subsection_lower:
                    initial_score *= 1.2

            # Boost if specific methodological terms appear in the chunk content
            for term in query_methodological_terms:
                if term in content_lower:
                    initial_score *= 1.05

            enhanced_result['similarity_score'] = initial_score
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _rank_results(self, results: List[Dict[str, Any]], query_analysis: Dict[str, Any], query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """  Final ranking of results, potentially using a cross-encoder for re-scoring. Applies diversity filtering.  """

        # First sort by the enhanced similarity score
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        diverse_results = []
        seen_document_ids = set()
        seen_chunk_ids = set()
        seen_section_titles = set()

        for result in results:
            doc_id = result.get('document_id')
            chunk_id = result.get('chunk_id')
            section_title = result.get('section_title')

            if chunk_id in seen_chunk_ids:
                continue

            if doc_id not in seen_document_ids or \
                (section_title and (section_title not in seen_section_titles or sum(1 for r in diverse_results if r.get('section_title') == section_title) < 2)):
                diverse_results.append(result)
                seen_document_ids.add(doc_id)
                seen_chunk_ids.add(chunk_id)
                if section_title: seen_section_titles.add(section_title)
            else:
                # Fallback: allow a few highly relevant chunks from already seen documents/sections if they are top-ranked
                max_score_for_doc = next((d['similarity_score'] for d in diverse_results if d['document_id'] == doc_id), 0)
                chunks_from_this_doc_in_diverse = sum(1 for r in diverse_results if r['document_id'] == doc_id)

                if result.get('similarity_score', 0) >= 0.9 * max_score_for_doc and chunks_from_this_doc_in_diverse < 3:
                    diverse_results.append(result)
                    seen_chunk_ids.add(chunk_id)

        # Ensure top_k is met from the overall ranked list if diverse_results are not enough
        final_selected_results = []
        final_seen_chunk_ids = set()

        for r in diverse_results:
            if r['chunk_id'] not in final_seen_chunk_ids:
                final_selected_results.append(r)
                final_seen_chunk_ids.add(r['chunk_id'])

        # Fill up to top_k with remaining highly-ranked chunks if necessary
        for r in results:
            if len(final_selected_results) >= top_k: # Cap at top_k
                break
            if r['chunk_id'] not in final_seen_chunk_ids:
                final_selected_results.append(r)
                final_seen_chunk_ids.add(r['chunk_id'])

        # Final sort after diversity for good measure
        final_selected_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return final_selected_results


# 8. RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.llm_model = LLM_MODEL_NAME
        self.tokenizer = tiktoken.encoding_for_model(self.llm_model)

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response from retrieved information using an LLM."""

        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information to answer your query.",
                'confidence': 0.0,
                'sources': []
            }

        # Prepare context for the LLM
        full_context_parts = []
        sources_for_output = []
        seen_source_documents_pages_sections = set() # NEW: To track unique combinations for sources summary

        current_context_tokens = 0
        prompt_overhead_tokens = len(self.tokenizer.encode("You are an intelligent assistant designed to provide answers based only on the provided information. Do not use any outside knowledge. If the answer cannot be found in the provided context, state that you cannot answer the question based on the given information. For each statement in your answer, indicate which 'Source Document ID' it came from. If a piece of information is found in multiple sources, you may cite all relevant sources. --- Information: --- Question: ")) + \
                                len(self.tokenizer.encode(query))

        available_tokens_for_context = LLM_MAX_TOKENS_CONTEXT - LLM_MAX_TOKENS_RESPONSE - prompt_overhead_tokens

        if available_tokens_for_context <= 0:
            logger.warning(f"Not enough tokens available for context after reserving for response and prompt overhead. Available: {available_tokens_for_context}")
            return {
                'answer': "System token limits prevent generating a full response. Try a shorter query.",
                'confidence': 0.0,
                'sources': []
            }

        # Iterating through retrieved chunks to build context
        for i, chunk in enumerate(retrieved_chunks):
            chunk_content = chunk['content']
            chunk_document_id = chunk['document_id']
            chunk_section = chunk.get('section_title', 'N/A')
            chunk_subsection = chunk.get('subsection_title', 'N/A')
            chunk_page = chunk.get('page_number', 'N/A')

            # MODIFIED: Format the chunk content for LLM context, including new metadata
            formatted_chunk = (
                f"Source Document ID: {chunk_document_id}\n"
                f"Page: {chunk_page}\n"
                f"Section: {chunk_section}\n"
                f"Subsection: {chunk_subsection}\n"
                f"Chunk Content:\n{chunk_content}\n---"
            )
            chunk_tokens = len(self.tokenizer.encode(formatted_chunk))

            if current_context_tokens + chunk_tokens < available_tokens_for_context:
                full_context_parts.append(formatted_chunk)
                current_context_tokens += chunk_tokens

                # MODIFIED: Prepare sources for the final output, adding new metadata for more context
                # Deduplicate sources based on document, page, and section for clearer output
                source_key = (chunk_document_id, chunk_page, chunk_section)
                if source_key not in seen_source_documents_pages_sections:
                    sources_for_output.append({
                        'document_id': chunk_document_id,
                        'score': chunk.get('similarity_score', 0),
                        'entities_in_chunk': [e.get('name', '') for e in chunk.get('entities', [])],
                        'keywords_in_chunk': chunk.get('keywords', []),
                        'page_number': chunk_page,
                        'section_title': chunk_section,
                        'subsection_title': chunk_subsection
                    })
                    seen_source_documents_pages_sections.add(source_key)
            else:
                logger.info(f"Stopped adding chunks due to token limit. Added {i} chunks. Total context tokens: {current_context_tokens}")
                break

        full_context = "\n\n".join(full_context_parts)

        # Define the prompt for the LLM
        # --- MODIFIED SYSTEM PROMPT ---
        prompt = f"""
                    You are an intelligent, expert hydrological assistant. Your primary directive is to provide concise, accurate, and comprehensive answers *strictly* based *only* on the provided 'Information' below.
                    Do not use any outside knowledge or information not explicitly present in the given context.
                    
                    **IMPORTANT GUIDELINES:**
                    1. For *every* statement or piece of factual content in your answer, you *must* indicate its 'Source Document ID' using the format '[Source Document ID: X]' directly after the relevant text. If a piece of information is derived from multiple provided sources, cite all applicable 'Source Document IDs'.
                    2. If the answer to a part of the user's question cannot be found or fully inferred from the provided 'Information', clearly state: "I cannot answer this specific detail definitively based on the provided information."
                    3. Address ALL components of the user's question comprehensively and distinctly. If the question asks for multiple things (e.g., "how X, what Y, and why Z?"), ensure each part is answered in a structured manner (e.g., using bullet points or numbered lists for each sub-question).
                    4. Pay extremely close attention to precise definitions, numerical values, and the *exact purpose and function* of methodologies described in the context. For example, differentiate clearly between different types of thresholds or methods and their intended outcomes (e.g., "MST" for independence of RI events vs. "double-threshold approach" for filtering influential MHWs).
                    5. When providing numerical values or specific criteria, quote them precisely as found in the source.

                    ---
                    Information:
                    {full_context}
                    ---

                    Question: {query}
                    """
        # --- END MODIFIED SYSTEM PROMPT ---

        llm_answer = "An error occurred while generating the response."
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Keep temperature low for factual accuracy
                max_tokens=LLM_MAX_TOKENS_RESPONSE
            )
            llm_answer = chat_completion.choices[0].message.content.strip()
            logger.info("LLM response generated.")
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            llm_answer = f"An error occurred with the OpenAI API: {e}. Please check your API key and network."
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            llm_answer = f"An unexpected error occurred during response generation: {e}."

        # Calculate confidence - simple average of top chunk scores
        avg_score = sum(chunk.get('similarity_score', 0) for chunk in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0
        confidence = min(avg_score, 1.0) # Ensure it doesn't exceed 1.0

        return {
            'answer': llm_answer,
            'confidence': confidence,
            'sources': sources_for_output,
            'context_used_chunks_count': len(full_context_parts)
        }


# 9. PERSISTENCE MANAGER
# =============================================================================

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


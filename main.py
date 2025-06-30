import pypdf
import asyncio
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import uuid
import logging
import re
import os
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
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
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
KG_GRAPH_PATH = STORAGE_PATH / "kg_graph.gml" # GML for NetworkX graph
KG_ENTITIES_PATH = STORAGE_PATH / "kg_entities.json"
KG_RELATIONS_PATH = STORAGE_PATH / "kg_relations.json"
LAST_DATA_HASH_PATH = STORAGE_PATH / "data_hash.txt"
KG_CHUNKS_PATH = STORAGE_PATH / "kg_chunks.json"

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

    def to_dict(self):
        return {
            'id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'chunk_index': self.chunk_index,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'keywords': self.keywords,
            'entities': self.entities,
            'metadata': self.metadata
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
            metadata=data['metadata']
        )

@dataclass
class KnowledgeEntity:
    id: str
    name: str
    entity_type: str
    confidence: float
    document_ids: List[str] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)

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
    """Service for generating text embeddings. Can use OpenAI or Sentence Transformers."""
    def __init__(self, api_key: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Determine whether to use OpenAI API or a local Sentence Transformer model
        if model_name.startswith("text-embedding-") or model_name.startswith("gpt-"):
            if not api_key:
                raise ValueError(f"OPENAI_API_KEY must be set for OpenAI embedding model: {model_name}")
            self.client = OpenAI(api_key=api_key)
            self.model_type = "openai_api"
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            self.EMBEDDING_MODEL_MAX_TOKENS = 8192 # OpenAI Ada-002 limit
            logger.info(f"EmbeddingService initialized with OpenAI API model: {model_name}")
        else: # Assume it's a local Sentence Transformer model
            try:
                self.model = SentenceTransformer(model_name)
                self.model_type = "sentence_transformer"
                # For local models, tiktoken is only used for LLM side, not for ST's internal tokenization.
                self.tokenizer = tiktoken.encoding_for_model(LLM_MODEL_NAME) # Use LLM tokenizer for generic token counting
                self.EMBEDDING_MODEL_MAX_TOKENS = self.model.max_seq_length # Use the actual model's max sequence length
                if self.EMBEDDING_MODEL_MAX_TOKENS is None:
                    # Fallback if model doesn't expose max_seq_length, choose a conservative default
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
                    # Attempt to reshape/pad if dimensions are off (though better to have correct config)
                    if embedding.shape[0] < self.embedding_dim:
                        padded_embedding = np.pad(embedding, (0, self.embedding_dim - embedding.shape[0]), 'constant')
                        return padded_embedding.astype(np.float32)
                    elif embedding.shape[0] > self.embedding_dim:
                        return embedding[:self.embedding_dim].astype(np.float32) # Truncate if too long
                    
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
        """Process document into chunks with entities and relations"""

        logger.info(f"Processing document: {document_id}")
        cleaned_content = self._clean_text(content)

        # Extract entities first
        entities = self._extract_entities(cleaned_content, document_id)

        chunks = self._create_chunks(cleaned_content, document_id)

        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = self.embedding_service.get_embedding(chunk.content)

        # Extract keywords
        self._extract_keywords(chunks)

        # Link entities to chunks
        self._link_entities_to_chunks(chunks, entities)

        # Create relations
        relations = self._create_relations(entities, cleaned_content, document_id)

        return chunks, entities, relations

    def _clean_text(self, content: str) -> str:
        # Initial cleanup: Remove non-ASCII and normalize whitespace aggressively
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        content = re.sub(r'\s+', ' ', content)

        # Remove common PDF internal artifacts

        # Regex for common PDF object/stream identifiers (e.g., "0000000015 00000 n")
        content = re.sub(r'\b\d+\s+\d+\s+n\b|\b\d{9,}\s+\d{5,}\b', ' ', content)
        # Remove timestamps (e.g., "D:20250619015147Z")
        content = re.sub(r'D:\d{14}Z', ' ', content)
        # Remove common page number patterns (e.g., "1 of 10", or just isolated page numbers like "2 4")
        content = re.sub(r'\bPage \d+ of \d+\b|\b\d+\s+\d+\b', ' ', content, flags=re.IGNORECASE)

        # Remove very short sequences that look like OCR errors or PDF junk (e.g., "kI", "jE", "OV")

        # This is more precise: look for 2-3 char sequences that are NOT common English words
        content = re.sub(r'\b[A-Za-z]{2,3}\b', ' ', content) # Removes short words, some might be valid.
        # Remove reference numbers like "[18][19]" or just standalone numbers that are unlikely to be meaningful entities.
        content = re.sub(r'\[\d+\](\[\d+\])*', ' ', content) # Removes [18][19], [1], etc.
        content = re.sub(r'\b\d+\.\d+\b', ' ', content) # Removes "2.2", "3.2", "4.3"
        # After targeted removal, normalize whitespace again to clean up extra spaces left by substitutions
        content = re.sub(r'\s+', ' ', content)

        return content.strip()

    def _create_chunks(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """ Create overlapping chunks based on a recursive splitting strategy, respecting token limits and minimum chunk size. """
        chunks = []
        chunk_index = 0

        # Use the EMBEDDING_MODEL_MAX_TOKENS from the embedding service instance
        # This allows it to dynamically adapt to the actual max_seq_length of the loaded ST model.
        EMBEDDING_MODEL_HARD_MAX_TOKENS = self.embedding_service.EMBEDDING_MODEL_MAX_TOKENS

        # A very safe fallback character split size, ensuring it's always smaller than the hard max.
        FALLBACK_CHAR_CHUNK_SIZE = int(EMBEDDING_MODEL_HARD_MAX_TOKENS * 0.9) # Leave a bit more buffer

        separators = ["\n\n", "\n", ". ", "? ", "! ", " "]

        def _split_recursively(text: str, current_separators: List[str]) -> List[str]:
            if not text:
                return []

            text_tokens = len(self.tokenizer.encode(text)) # Use LLM tokenizer for this count

            # Base case: if text fits within TARGET_CHUNK_SIZE_TOKENS, return it
            if text_tokens <= self.target_chunk_size_tokens:
                return [text]

            # If no more semantic separators, force split by character
            if not current_separators:
                # Break text into chunks of FALLBACK_CHAR_CHUNK_SIZE characters
                sub_chunks = []
                for i in range(0, len(text), FALLBACK_CHAR_CHUNK_SIZE):
                    sub_chunk = text[i:i + FALLBACK_CHAR_CHUNK_SIZE]
                    sub_chunks.append(sub_chunk)

                    # Log a warning if even after char splitting, a sub_chunk still exceeds the hard token limit
                    if len(self.tokenizer.encode(sub_chunk)) > EMBEDDING_MODEL_HARD_MAX_TOKENS:
                        logger.warning(f"Recursive char split produced a chunk of {len(self.tokenizer.encode(sub_chunk))} tokens, still exceeding {EMBEDDING_MODEL_HARD_MAX_TOKENS}. Document: {doc_id}. Consider reducing CHUNK_SIZE_TOKENS or checking source document. Snippet: {sub_chunk[:100]}...")
                return sub_chunks


            current_sep = current_separators[0]
            remaining_seps = current_separators[1:]

            split_parts = text.split(current_sep)

            collected_chunks_from_recursion = []
            current_group_of_parts = []
            current_group_tokens = 0

            for part in split_parts:
                part = part.strip()
                if not part:
                    continue

                part_tokens = len(self.tokenizer.encode(part))

                # If adding this part makes the current group too large,
                # process the current group and then handle the new part.
                if current_group_tokens + part_tokens > self.target_chunk_size_tokens:
                    if current_group_of_parts:
                        # Recursively split the collected parts.
                        collected_chunks_from_recursion.extend(_split_recursively(
                            current_sep.join(current_group_of_parts), remaining_seps
                        ))

                    # The current 'part' itself might be too large, so recursively split it.
                    collected_chunks_from_recursion.extend(_split_recursively(part, remaining_seps))

                    # Reset collector for the next group
                    current_group_of_parts = []
                    current_group_tokens = 0
                else:
                    # Add part to current group
                    current_group_of_parts.append(part)
                    current_group_tokens += part_tokens

            # Process any remaining parts in the last group
            if current_group_of_parts:
                collected_chunks_from_recursion.extend(_split_recursively(
                    current_sep.join(current_group_of_parts), remaining_seps
                ))

            return collected_chunks_from_recursion

        # Perform initial recursive split
        pre_overlap_chunks = _split_recursively(content, separators)

        # Now, combine chunks with overlap, ensuring final chunk size constraint
        current_overlap_sentences_buffer = []
        current_combined_chunk_content = []
        current_combined_chunk_tokens = 0

        # Attempt to make sentence-level overlaps from the pre_overlap_chunks
        for pre_chunk_content in pre_overlap_chunks:
            if self.nlp:
                doc_pre_chunk = self.nlp(pre_chunk_content)
                sentences_from_pre_chunk = [sent.text for sent in doc_pre_chunk.sents]
            else:
                sentences_from_pre_chunk = [s.strip() for s in pre_chunk_content.split('. ') if s.strip()]

            for sent in sentences_from_pre_chunk:
                sent_tokens = len(self.tokenizer.encode(sent))

                if current_combined_chunk_tokens + sent_tokens > self.target_chunk_size_tokens:
                    if current_combined_chunk_content: # Ensure there's content to save
                        final_chunk_text = " ".join(current_combined_chunk_content)
                        final_chunk_tokens = len(self.tokenizer.encode(final_chunk_text))

                        if final_chunk_tokens >= self.min_chunk_size_tokens or not chunks:
                            chunks.append(DocumentChunk(
                                id=str(uuid.uuid4()),
                                document_id=doc_id,
                                content=final_chunk_text,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1
                        else:
                            logger.debug(f"Skipping chunk {chunk_index} from {doc_id} due to small size: {final_chunk_tokens} tokens.")

                    current_combined_chunk_content = list(current_overlap_sentences_buffer)
                    current_combined_chunk_content.append(sent)
                    current_combined_chunk_tokens = len(self.tokenizer.encode(" ".join(current_combined_chunk_content)))
                    current_overlap_sentences_buffer = current_combined_chunk_content[-self.chunk_overlap_sentences:] 
                else:
                    current_combined_chunk_content.append(sent)
                    current_combined_chunk_tokens += sent_tokens
                    current_overlap_sentences_buffer = current_combined_chunk_content[-self.chunk_overlap_sentences:]


        # Add the very last remaining chunk
        if current_combined_chunk_content:
            final_chunk_text = " ".join(current_combined_chunk_content)
            final_chunk_tokens = len(self.tokenizer.encode(final_chunk_text))

            if final_chunk_tokens > 0:
                chunks.append(DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    content=final_chunk_text,
                    chunk_index=chunk_index
                ))
            else:
                logger.debug(f"Skipping final chunk from {doc_id} due to empty content.")

        logger.info(f"Document {doc_id} split into {len(chunks)} chunks.")
        return chunks

    def _extract_entities(self, content: str, doc_id: str) -> List[KnowledgeEntity]:
        """Extract named entities using spaCy NER and custom PhraseMatcher rules."""

        entities = []
        if not self.nlp:
            return entities

        doc = self.nlp(content)
        seen_entities = set()

        # 1. Add entities found by spaCy's default NER
        for ent in doc.ents:
            if len(ent.text.strip()) < 2 or ent.text.strip().isnumeric():
                continue
            entity_key = (ent.text.lower(), ent.label_)
            if entity_key not in seen_entities:
                entity = KnowledgeEntity(
                    id=str(uuid.uuid4()),
                    name=ent.text,
                    entity_type=ent.label_,
                    confidence=0.9,
                    document_ids=[doc_id]
                )
                entities.append(entity)
                seen_entities.add(entity_key)

        # 2. Add entities found by custom PhraseMatcher rules
        if self.phrase_matcher:
            matches = self.phrase_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                custom_entity_type = self.nlp.vocab.strings[match_id]

                if len(span.text.strip()) < 2 or span.text.strip().isnumeric():
                    continue

                entity_key = (span.text.lower(), custom_entity_type)
                if entity_key not in seen_entities:
                    entity = KnowledgeEntity(
                        id=str(uuid.uuid4()),
                        name=span.text,
                        entity_type=custom_entity_type,
                        confidence=0.95, # Higher confidence for direct phrase matches
                        document_ids=[doc_id]
                    )
                    entities.append(entity)
                    seen_entities.add(entity_key)

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

    def _create_relations(self, entities: List[KnowledgeEntity], full_content: str, doc_id: str) -> List[KnowledgeRelation]:
        """ Create relations using spaCy's dependency parser and rule-based patterns. Falls back to co-occurrence if patterns don't match. """

        relations = []
        if not self.nlp or not self.matcher:
            return relations

        doc = self.nlp(full_content)
        entity_map = {ent.name: ent for ent in entities}

        # Rule-based extraction using spaCy Matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            rule_name = self.nlp.vocab.strings[match_id]

            # Collect actual entities that are part of the original `entities` list
            span_entities_in_kg = []
            for ent_in_span in span.ents:
                if ent_in_span.text in entity_map:
                    span_entities_in_kg.append(entity_map[ent_in_span.text])

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
                if ent_in_sent.text in entity_map:
                    sent_entities.append(entity_map[ent_in_sent.text])

            if self.phrase_matcher:
                phrase_matches_in_sent = self.phrase_matcher(sent)
                for match_id, start, end in phrase_matches_in_sent:
                    span_in_sent = sent.char_span(start, end)
                    if span_in_sent and span_in_sent.text in entity_map: # Check if this phrase is also a known entity
                        if entity_map[span_in_sent.text] not in sent_entities:
                            sent_entities.append(entity_map[span_in_sent.text])

            if len(sent_entities) >= 2:
                for i, ent1 in enumerate(sent_entities):
                    for j, ent2 in enumerate(sent_entities):
                        if i < j: # Avoid self-loops and duplicate (A,B) and (B,A) pairs
                            # Check if a specific or CO_OCCURS relation already exists for this pair
                            if (ent1.id, ent2.id, 'CO_OCCURS') not in extracted_relations_set and (ent2.id, ent1.id, 'CO_OCCURS') not in extracted_relations_set:
                                # Check if a more specific relation already exists between these two entities
                                specific_relation_exists = False
                                for r_check in extracted_relations_set:
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
        return relations


# 4. VECTOR INDEX MANAGER
# =============================================================================

class VectorIndexManager:
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: Dict[str, Any] = {} # Stores serialized chunk data
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.current_index = 0
        self._initialize_index()

    def _initialize_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to vector index"""
        if not chunks:
            return

        embeddings_to_add = []
        for chunk in chunks:
            if chunk.embedding is not None and chunk.id not in self.id_to_index: # Only add new chunks
                embeddings_to_add.append(chunk.embedding)
                # Store metadata
                self.chunk_metadata[chunk.id] = chunk.to_dict()
                # Store mappings
                self.id_to_index[chunk.id] = self.current_index
                self.index_to_id[self.current_index] = chunk.id
                self.current_index += 1
            elif chunk.id in self.id_to_index:
                # Update existing chunk metadata if content might have changed (though in this flow, it's new docs)
                self.chunk_metadata[chunk.id] = chunk.to_dict()
        if embeddings_to_add:
            embeddings_array = np.array(embeddings_to_add).astype('float32')
            self.index.add(embeddings_array)
            logger.info(f"Added {len(embeddings_to_add)} new embeddings to FAISS index. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. No search results.")
            return []

        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.index_to_id:
                chunk_id = self.index_to_id[idx]
                chunk_dict = self.chunk_metadata.get(chunk_id, {})

                if chunk_dict:
                    # For search results, dict is fine to avoid unnecessary object creation
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
            self.graph.add_node(chunk.id, type='CHUNK', content=chunk.content[:200] + '...', document_id=chunk.document_id, chunk_index=chunk.chunk_index)

        for entity in entities:
            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, name=entity.name, type=entity.entity_type, document_ids=entity.document_ids, related_chunks=entity.related_chunks)
            else:
                existing_entity = self.entities[entity.id]
                existing_entity.document_ids.extend([d_id for d_id in entity.document_ids if d_id not in existing_entity.document_ids])
                existing_entity.related_chunks.extend([c_id for c_id in entity.related_chunks if c_id not in existing_entity.related_chunks])
                self.graph.nodes[entity.id]['name'] = entity.name
                self.graph.nodes[entity.id]['type'] = entity.entity_type
                self.graph.nodes[entity.id]['document_ids'] = existing_entity.document_ids
                self.graph.nodes[entity.id]['related_chunks'] = existing_entity.related_chunks

        for entity in entities:
            for chunk_id in entity.related_chunks:
                if chunk_id in self.graph and entity.id in self.graph:
                    if self.graph.get_edge_data(chunk_id, entity.id) is None:
                        self.graph.add_edge(chunk_id, entity.id, key='MENTIONS', relation_type='MENTIONS', confidence=1.0)
                    if self.graph.get_edge_data(entity.id, chunk_id) is None:
                        self.graph.add_edge(entity.id, chunk_id, key='MENTIONED_IN', relation_type='MENTIONED_IN', confidence=1.0)
                else:
                    logger.debug(f"Skipping MENTIONS relation: Chunk {chunk_id} or Entity {entity.id} not found in graph for direct linking.")

        # Add relations (between entities)
        added_relation_tuples = set()

        for relation in relations:
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

        # Find entity IDs from query names
        query_entity_ids = []
        for q_ent_name in query_entities:
            for eid, entity_obj in self.entities.items():
                if entity_obj.name.lower() == q_ent_name.lower():
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
                        else:
                            info_item['content_snippet'] = node_data.get('content', '')
                            info_item['document_id'] = "UNKNOWN"
                    elif node_data.get('type') in ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC', 'LOCATION', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'STUDY', 'WATER_BODY', 'HYDRO_MEASUREMENT', 'POLLUTANT', 'HYDRO_EVENT', 'HYDRO_INFRASTRUCTURE', 'HYDRO_MODEL']:
                        info_item['name'] = node_data.get('name')
                        if self.graph.has_edge(start_node_id, target_node):
                            connecting_edge_data = self.graph.get_edge_data(start_node_id, target_node)
                            if connecting_edge_data:
                                first_edge_key = list(connecting_edge_data.keys())[0]
                                edge_attrs = connecting_edge_data[first_edge_key] if isinstance(connecting_edge_data, dict) else connecting_edge_data
                                info_item['connection_type'] = edge_attrs.get('relation_type', 'unknown')
                                info_item['connection_confidence'] = edge_attrs.get('confidence', 0.0)
                    related_info.append(info_item)
                    seen_nodes.add(target_node)
        return related_info

    def save_graph(self) -> bool:
        """Saves the NetworkX graph and associated entity/relation data."""
        try:
            nx.write_gml(self.graph, str(KG_GRAPH_PATH))

            with open(KG_ENTITIES_PATH, 'w', encoding='utf-8') as f:
                json.dump({eid: entity.to_dict() for eid, entity in self.entities.items()}, f, indent=2)
            with open(KG_RELATIONS_PATH, 'w', encoding='utf-8') as f:
                json.dump({rid: relation.to_dict() for rid, relation in self.relations.items()}, f, indent=2)

            # Save document chunks (metadata only, no embeddings)
            chunks_to_save = {cid: chunk.to_dict() for cid, chunk in self.document_chunks.items()}
            for cid in chunks_to_save:
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
                self.graph = nx.DiGraph()
                return False
        logger.info("No existing Knowledge Graph found. Starting fresh.")
        return False



# 6. QUERY PROCESSOR
# =============================================================================

class QueryProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load(SPACY_MODEL) # Use SPACY_MODEL from .env
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
        """Analyze query to understand intent and extract entities/keywords."""

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

        extracted_entity_names = [e['text'] for e in entities if e['label'] in ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC', 'STUDY', 'WATER_BODY', 'HYDRO_MEASUREMENT', 'POLLUTANT', 'HYDRO_EVENT', 'HYDRO_INFRASTRUCTURE', 'HYDRO_MODEL']]

        return {
            'original_query': query,
            'query_type': query_type,
            'keywords': keywords,
            'entities': extracted_entity_names,
            'complexity': 'simple' if len(query.split()) < 5 else 'medium' if len(query.split()) < 15 else 'complex'
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

        query_embedding = self.embedding_service.get_embedding(query)

        # 1. Initial Vector Search (Retrieve more candidates than final top_k)
        vector_candidates = self.vector_index.search(query_embedding, top_k * 5)

        # 2. Knowledge Graph Enhancement
        kg_related_info = self.kg_manager.get_related_info(query_analysis.get('entities', []))

        # Boost vector results based on KG relatedness and keyword overlap
        enhanced_candidates = self._enhance_and_boost_candidates(vector_candidates, query_analysis, kg_related_info)

        # Add KG-derived chunks as candidates if they are not already in vector_candidates
        kg_chunk_candidates = []
        existing_candidate_chunk_ids = {c['chunk_id'] for c in enhanced_candidates}

        for item in kg_related_info:
            if item['type'] == 'CHUNK':
                chunk_id = item['id']
                if chunk_id not in existing_candidate_chunk_ids:
                    chunk_data = self.vector_index.chunk_metadata.get(chunk_id)
                    if chunk_data:
                        # Assign a score based on KG distance, lower distance = higher score; 1.0 for distance 1, 0.5 for distance 2 etc.
                        kg_score = max(0.1, 1.0 - (item['distance_from_query_entity'] * 0.2))
                        kg_chunk_candidates.append({
                            'chunk_id': chunk_id,
                            'similarity_score': kg_score,
                            'content': chunk_data.get('content', ''),
                            'document_id': chunk_data.get('document_id', ''),
                            'keywords': chunk_data.get('keywords', []),
                            'entities': chunk_data.get('entities', [])
                        })
                        existing_candidate_chunk_ids.add(chunk_id)

        enhanced_candidates.extend(kg_chunk_candidates)

        # 3. Apply Re-ranking (Cross-encoder or RRF)
        final_results = self._rank_results(enhanced_candidates, query_analysis, query)

        return final_results[:top_k]

    def _enhance_and_boost_candidates(self, candidates: List[Dict[str, Any]], query_analysis: Dict[str, Any], kg_related_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Enhance and boost scores of candidates based on KG relatedness and keyword overlap. """

        enhanced_results = []
        query_entities = [e.lower() for e in query_analysis.get('entities', [])]
        query_keywords = [kw.lower() for kw in query_analysis.get('keywords', [])]

        for result in candidates:
            enhanced_result = result.copy()
            initial_score = enhanced_result.get('similarity_score', 0)

            # --- Boost based on KG entity overlap ---
            doc_entities = [e.get('name', '').lower() for e in result.get('entities', [])]
            entity_overlap_count = len(set(query_entities) & set(doc_entities))
            if entity_overlap_count > 0:
                initial_score *= (1 + entity_overlap_count * 0.2)

            # --- Boost based on direct keyword matches ---
            content_lower = result.get('content', '').lower()
            doc_keywords = [kw.lower() for kw in result.get('keywords', [])]

            keyword_content_matches = sum(1 for kw in query_keywords if kw in content_lower)
            keyword_tag_matches = len(set(query_keywords) & set(doc_keywords))
            total_keyword_matches = keyword_content_matches + keyword_tag_matches

            if total_keyword_matches > 0:
                initial_score *= (1 + total_keyword_matches * 0.05)

            # --- Boost based on KG related chunks/entities (if a chunk itself is KG-related) ---
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

            enhanced_result['similarity_score'] = initial_score
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _rank_results(self, results: List[Dict[str, Any]], query_analysis: Dict[str, Any], query_text: str) -> List[Dict[str, Any]]:
        """  Final ranking of results, potentially using a cross-encoder for re-scoring. Applies diversity filtering.  """

        # First sort by the enhanced similarity score
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        # Cross-encoder Re-ranking (if available) - Currently mocked
        if self.cross_encoder and len(results) > 1:
            sentence_pairs = [(query_text, r['content']) for r in results]
            try:
                # actual_scores = self.cross_encoder.predict(sentence_pairs)
                actual_scores = np.random.rand(len(results)) * 0.5 + 0.5
                for i, score in enumerate(actual_scores):
                    results[i]['similarity_score'] = (results[i]['similarity_score'] + score) / 2
                results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                logger.info("Results re-ranked by cross-encoder (mocked).")
            except Exception as e:
                logger.warning(f"Cross-encoder re-ranking failed: {e}. Falling back to similarity score.")

        # Apply diversity filter to ensure variety of source documents
        diverse_results = []
        seen_document_ids = set()
        seen_chunk_ids = set()

        for result in results:
            doc_id = result.get('document_id')
            chunk_id = result.get('chunk_id')

            if chunk_id in seen_chunk_ids:
                continue

            if doc_id not in seen_document_ids:
                diverse_results.append(result)
                seen_document_ids.add(doc_id)
                seen_chunk_ids.add(chunk_id)
            else:
                # Find the max score for this document already in diverse_results
                max_score_for_doc = next((d['similarity_score'] for d in diverse_results if d['document_id'] == doc_id), 0)
                chunks_from_this_doc_in_diverse = sum(1 for r in diverse_results if r['document_id'] == doc_id)

                if result.get('similarity_score', 0) >= 0.9 * max_score_for_doc and chunks_from_this_doc_in_diverse < 2:
                    diverse_results.append(result)
                    seen_chunk_ids.add(chunk_id)

        # This fills up the ranks using the initial sorted list if the diverse_results fall short of top_k.
        final_selected_results = []
        final_seen_chunk_ids = set()

        # Add diverse results first
        for r in diverse_results:
            if r['chunk_id'] not in final_seen_chunk_ids:
                final_selected_results.append(r)
                final_seen_chunk_ids.add(r['chunk_id'])

        # This ensures we meet `top_k` if enough candidates exist.
        for r in results:
            if len(final_selected_results) >= len(results):
                break
            if r['chunk_id'] not in final_seen_chunk_ids:
                final_selected_results.append(r)
                final_seen_chunk_ids.add(r['chunk_id'])

        # Final sort after diversity for good measure (important for RRF type merging if applied)
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
        context_parts = []
        sources_for_output = []
        seen_source_documents = set() # To track unique documents for citations

        current_context_tokens = 0
        # This is a rough estimate; true overhead depends on model and prompt structure
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

        for i, chunk in enumerate(retrieved_chunks):
            chunk_content = chunk['content']
            chunk_document_id = chunk['document_id']

            # Format the chunk content for LLM context
            formatted_chunk = f"Source Document ID: {chunk_document_id}\nChunk Content:\n{chunk_content}\n---"
            chunk_tokens = len(self.tokenizer.encode(formatted_chunk))

            if current_context_tokens + chunk_tokens < available_tokens_for_context:
                context_parts.append(formatted_chunk)
                current_context_tokens += chunk_tokens

                # Prepare sources for the final output
                if chunk_document_id not in seen_source_documents:
                    sources_for_output.append({
                        'document_id': chunk.get('document_id', 'N/A'),
                        'score': chunk.get('similarity_score', 0),
                        'entities_in_chunk': [e.get('name', '') for e in chunk.get('entities', [])],
                        'keywords_in_chunk': chunk.get('keywords', [])
                    })
                    seen_source_documents.add(chunk_document_id)
            else:
                logger.info(f"Stopped adding chunks due to token limit. Added {i} chunks. Total context tokens: {current_context_tokens}")
                break

        full_context = "\n\n".join(context_parts)

        # Define the prompt for the LLM
        prompt = f"""
                    You are an intelligent assistant designed to provide answers based *only* on the provided information.
                    Do not use any outside knowledge. If the answer cannot be found in the provided context, state that you cannot answer the question based on the given information.
                    For each statement in your answer, indicate which "Source Document ID" it came from.
                    If a piece of information is found in multiple sources, you may cite all relevant sources.

                    ---
                    Information:
                    {full_context}
                    ---

                    Question: {query}
                    """

        llm_answer = "An error occurred while generating the response."
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
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
            'context_used_chunks_count': len(context_parts)
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


# 10. MAIN RAG SYSTEM
# =============================================================================

class SimpleAdvancedRAGSystem:
    def __init__(self):
        # Initialize shared embedding service
        # Pass OPENAI_API_KEY as an argument, even if it might be None for ST models
        self.embedding_service = EmbeddingService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBEDDING_MODEL_NAME, # This will be 'all-MiniLM-L6-v2' from .env
            embedding_dim=EMBEDDING_DIM      # This will be '384' from .env
        )

        self.storage_manager = StorageManager(DATA_PATH, STORAGE_PATH)
        self.doc_processor = DocumentProcessor(self.embedding_service) # Pass embedding service
        self.vector_index = VectorIndexManager(embedding_dim=EMBEDDING_DIM) # Ensure embedding_dim matches chosen model
        self.kg_manager = KnowledgeGraphManager()
        self.retrieval_engine = HybridRetrievalEngine(self.vector_index, self.kg_manager, self.embedding_service) # Pass embedding service
        self.response_generator = ResponseGenerator() # This still uses OpenAI's LLM, so it will still need the API key

        logger.info("RAG System components initialized.")

    def initialize_knowledge_base(self):
        """Initializes or re-processes the knowledge base based on data changes."""
        # Attempt to load existing data
        if self.storage_manager.load_all(self.vector_index, self.kg_manager):
            # Data loaded, now check if original data files have changed
            if not self.storage_manager.is_data_changed():
                logger.info("Knowledge base is up-to-date. No re-processing needed.")
                return

        logger.info("Re-processing documents due to data changes or no prior data found.")
        # Clear existing data structures before re-processing
        self.vector_index._initialize_index() 
        self.vector_index.chunk_metadata = {}
        self.vector_index.id_to_index = {}
        self.vector_index.index_to_id = {}
        self.vector_index.current_index = 0

        self.kg_manager.graph = nx.MultiDiGraph()
        self.kg_manager.entities = {}
        self.kg_manager.relations = {}
        self.kg_manager.document_chunks = {}

        self._process_and_store_documents()
        self.storage_manager.save_all(self.vector_index, self.kg_manager)
        logger.info("Knowledge base re-processing complete and saved.")

        # Always regenerate the knowledge gap report after (re)initialization
        import knowledge_graph
        try:
            knowledge_graph.analyze_rag_knowledge_graph(self)
            print("Knowledge gap report saved to knowledge_gap_report.md and printed above.")
        except Exception as e:
            logger.error(f"Error during knowledge graph visualization/analysis: {e}")

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
        for file_path in directory.iterdir():
            if file_path.is_file():
                content = ""
                if file_path.suffix.lower() == '.pdf':
                    try:
                        reader = pypdf.PdfReader(file_path)
                        for page in reader.pages:
                            # Extract text from each page and concatenate
                            content += page.extract_text(0) + "\n"
                        logger.info(f"Loaded PDF document: {file_path.name}")
                        # logger.info(f"PDF EXTRACTED TEXT (first 500 chars of {file_path.name}):\n{content[:500]}\n...")
                    except Exception as e:
                        logger.error(f"Could not read PDF {file_path} using pypdf: {e}")
                        continue # Skip to next file if there's an error
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

                if content: # Only append if content was successfully extracted
                    documents.append({'id': file_path.name, 'content': content})
        return documents

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query and return results."""
        logger.info(f"Processing query: '{query_text}'")
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieval_engine.retrieve(query_text, top_k)

            # Analyze query for response generation (can reuse analysis from retrieval engine if desired)
            query_analysis = self.retrieval_engine.query_processor.analyze_query(query_text)

            # Generate response
            response = self.response_generator.generate_response(
                query_text, retrieved_chunks, query_analysis
            )

            # Add metadata
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


# 11. USAGE EXAMPLE
# =============================================================================

# Global variable to cache the RAG system instance
_rag_system = None

def answer_query(query: str) -> str:
    global _rag_system
    if _rag_system is None:
        _rag_system = SimpleAdvancedRAGSystem()
        _rag_system.initialize_knowledge_base()
    result = _rag_system.query(query)
    if result.get('status') == 'success':
        return result.get('answer', 'No answer found.')
    else:
        return f"Error: {result.get('error', 'Unknown error')}"
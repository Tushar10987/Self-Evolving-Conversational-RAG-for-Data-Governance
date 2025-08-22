"""
Multi-modal retrieval system for data governance queries.
Implements dense (FAISS), sparse (BM25), and metadata-based retrievers with ensemble scoring.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import logging

# Vector store and retrieval
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# LangChain components
from langchain.schema import Document
try:
    # Newer langchain moved Embeddings to langchain_core
    from langchain_core.embeddings import Embeddings as LCEmbeddings
except Exception:  # pragma: no cover
    try:
        from langchain.embeddings.base import Embeddings as LCEmbeddings  # type: ignore
    except Exception:
        LCEmbeddings = object  # fallback for typing only
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from config import retriever_config, data_config, system_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from a single retriever."""
    documents: List[Document]
    scores: List[float]
    retriever_type: str
    retrieval_time: float

@dataclass
class EnsembleResult:
    """Result from ensemble retrieval."""
    documents: List[Document]
    scores: List[float]
    retriever_contributions: Dict[str, List[float]]
    total_time: float

class DenseRetriever:
    """Dense retriever using FAISS and sentence transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or retriever_config.DENSE_MODEL
        self.encoder = SentenceTransformer(self.model_name)
        self.vectorstore = None
        self.documents = []
        self._embedding_wrapper = None
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the dense retriever."""
        self.documents = documents
        
        # Create LangChain documents
        docs = []
        for doc in documents:
            langchain_doc = Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
            docs.append(langchain_doc)
        
        # Create FAISS vector store with proper API usage
        texts = [doc["content"] for doc in documents]
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Prepare pairs (text, embedding) as expected by langchain FAISS.from_embeddings
        text_embedding_pairs = list(zip(texts, embeddings.tolist()))

        # Minimal embeddings wrapper implementing required interface for queries
        class _SentenceTransformerEmbeddings(LCEmbeddings):  # type: ignore
            def __init__(self, encoder: SentenceTransformer):
                self._encoder = encoder

            def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore
                return self._encoder.encode(
                    texts,
                    convert_to_numpy=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            def embed_query(self, text: str) -> List[float]:  # type: ignore
                return self._encoder.encode(
                    [text],
                    convert_to_numpy=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]

        self._embedding_wrapper = _SentenceTransformerEmbeddings(self.encoder)

        self.vectorstore = FAISS.from_embeddings(
            text_embedding_pairs,
            embedding=self._embedding_wrapper,
            metadatas=[d["metadata"] for d in documents],
        )
        
        logger.info(f"Added {len(documents)} documents to dense retriever")
    
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """Retrieve documents using dense similarity."""
        start_time = time.time()
        
        top_k = top_k or retriever_config.DENSE_TOP_K
        
        if self.vectorstore is None:
            return RetrievalResult([], [], "dense", 0.0)
        
        # Use LangChain's similarity search
        try:
            search_results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            # Extract documents and scores
            docs = []
            doc_scores = []
            
            for doc, score in search_results:
                docs.append(doc)
                doc_scores.append(float(score))
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            # Fallback to simple search
            docs = self.vectorstore.similarity_search(query, k=top_k)
            doc_scores = [1.0] * len(docs)  # Default scores
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(docs, doc_scores, "dense", retrieval_time)

class SparseRetriever:
    """Sparse retriever using BM25."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the sparse retriever."""
        self.documents = documents
        
        # Prepare documents for BM25
        texts = [doc["content"] for doc in documents]
        tokenized_texts = [text.lower().split() for text in texts]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_texts)
        
        logger.info(f"Added {len(documents)} documents to sparse retriever")
    
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """Retrieve documents using BM25."""
        start_time = time.time()
        
        top_k = top_k or retriever_config.SPARSE_TOP_K
        
        if self.bm25 is None:
            return RetrievalResult([], [], "sparse", 0.0)
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        docs = []
        doc_scores = []
        
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc = self.documents[idx]
                langchain_doc = Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
                docs.append(langchain_doc)
                doc_scores.append(float(scores[idx]))
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(docs, doc_scores, "sparse", retrieval_time)

class MetadataRetriever:
    """Metadata-based retriever using graph-like relationships."""
    
    def __init__(self):
        self.documents = []
        self.metadata_index = {}
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the metadata retriever."""
        self.documents = documents
        
        # Build metadata index
        for i, doc in enumerate(documents):
            metadata = doc["metadata"]
            
            # Index by type
            doc_type = metadata.get("type", "unknown")
            if doc_type not in self.metadata_index:
                self.metadata_index[doc_type] = []
            self.metadata_index[doc_type].append(i)
            
            # Index by table_id if present
            if "table_id" in metadata:
                table_id = metadata["table_id"]
                if table_id not in self.metadata_index:
                    self.metadata_index[table_id] = []
                self.metadata_index[table_id].append(i)
            
            # Index by owner_id if present
            if "owner_id" in metadata:
                owner_id = metadata["owner_id"]
                if owner_id not in self.metadata_index:
                    self.metadata_index[owner_id] = []
                self.metadata_index[owner_id].append(i)
        
        logger.info(f"Added {len(documents)} documents to metadata retriever")
    
    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """Retrieve documents using metadata matching."""
        start_time = time.time()
        
        top_k = top_k or retriever_config.METADATA_TOP_K
        
        # Extract potential metadata terms from query
        query_lower = query.lower()
        
        # Find matching documents based on metadata
        matching_docs = []
        doc_scores = []
        
        for i, doc in enumerate(self.documents):
            metadata = doc["metadata"]
            score = 0.0
            
            # Score based on type matching
            doc_type = metadata.get("type", "")
            if doc_type in query_lower:
                score += 0.5
            
            # Score based on table name matching
            if "table_name" in metadata:
                table_name = metadata["table_name"].lower()
                if table_name in query_lower:
                    score += 1.0
            
            # Score based on owner name matching
            if "owner_name" in metadata:
                owner_name = metadata["owner_name"].lower()
                if owner_name in query_lower:
                    score += 0.8
            
            # Score based on department matching
            if "owner_department" in metadata:
                department = metadata["owner_department"].lower()
                if department in query_lower:
                    score += 0.6
            
            # Score based on transformation type
            if "transformation_type" in metadata:
                transform_type = metadata["transformation_type"].lower()
                if transform_type in query_lower:
                    score += 0.4
            
            if score > 0:
                langchain_doc = Document(
                    page_content=doc["content"],
                    metadata=metadata
                )
                matching_docs.append(langchain_doc)
                doc_scores.append(score)
        
        # Sort by score and take top-k
        sorted_pairs = sorted(zip(matching_docs, doc_scores), key=lambda x: x[1], reverse=True)
        docs, scores = zip(*sorted_pairs[:top_k]) if sorted_pairs else ([], [])
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(list(docs), list(scores), "metadata", retrieval_time)

class EnsembleRetriever:
    """Ensemble retriever that combines multiple retrieval strategies."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "dense": retriever_config.DENSE_WEIGHT,
            "sparse": retriever_config.SPARSE_WEIGHT,
            "metadata": retriever_config.METADATA_WEIGHT
        }
        
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.metadata_retriever = MetadataRetriever()
        
        self.documents = []
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to all retrievers."""
        self.documents = documents
        
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
        self.metadata_retriever.add_documents(documents)
        
        logger.info(f"Added {len(documents)} documents to ensemble retriever")
    
    def retrieve(self, query: str, top_k: int = None) -> EnsembleResult:
        """Retrieve documents using ensemble of retrievers."""
        start_time = time.time()
        
        top_k = top_k or retriever_config.ENSEMBLE_TOP_K
        
        # Get results from each retriever
        dense_result = self.dense_retriever.retrieve(query)
        sparse_result = self.sparse_retriever.retrieve(query)
        metadata_result = self.metadata_retriever.retrieve(query)
        
        # Combine results
        all_docs = {}
        retriever_contributions = {
            "dense": [],
            "sparse": [],
            "metadata": []
        }
        
        # Process dense results
        for doc, score in zip(dense_result.documents, dense_result.scores):
            doc_id = doc.page_content[:100]  # Use content as ID
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "doc": doc,
                    "scores": {"dense": 0.0, "sparse": 0.0, "metadata": 0.0}
                }
            all_docs[doc_id]["scores"]["dense"] = score
            retriever_contributions["dense"].append(score)
        
        # Process sparse results
        for doc, score in zip(sparse_result.documents, sparse_result.scores):
            doc_id = doc.page_content[:100]
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "doc": doc,
                    "scores": {"dense": 0.0, "sparse": 0.0, "metadata": 0.0}
                }
            all_docs[doc_id]["scores"]["sparse"] = score
            retriever_contributions["sparse"].append(score)
        
        # Process metadata results
        for doc, score in zip(metadata_result.documents, metadata_result.scores):
            doc_id = doc.page_content[:100]
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "doc": doc,
                    "scores": {"dense": 0.0, "sparse": 0.0, "metadata": 0.0}
                }
            all_docs[doc_id]["scores"]["metadata"] = score
            retriever_contributions["metadata"].append(score)
        
        # Calculate ensemble scores
        ensemble_scores = []
        ensemble_docs = []
        
        for doc_id, doc_info in all_docs.items():
            # Weighted ensemble score
            ensemble_score = (
                doc_info["scores"]["dense"] * self.weights["dense"] +
                doc_info["scores"]["sparse"] * self.weights["sparse"] +
                doc_info["scores"]["metadata"] * self.weights["metadata"]
            )
            
            if ensemble_score >= retriever_config.MIN_SCORE_THRESHOLD:
                ensemble_scores.append(ensemble_score)
                ensemble_docs.append(doc_info["doc"])
        
        # Sort by ensemble score
        sorted_pairs = sorted(zip(ensemble_docs, ensemble_scores), key=lambda x: x[1], reverse=True)
        final_docs, final_scores = zip(*sorted_pairs[:top_k]) if sorted_pairs else ([], [])
        
        total_time = time.time() - start_time
        
        return EnsembleResult(
            documents=list(final_docs),
            scores=list(final_scores),
            retriever_contributions=retriever_contributions,
            total_time=total_time
        )
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update retriever weights."""
        self.weights = new_weights
        logger.info(f"Updated ensemble weights: {new_weights}")

def load_documents() -> List[Dict[str, Any]]:
    """Load documents from the data directory."""
    documents_file = Path(data_config.DATA_DIR) / "documents.json"
    
    if not documents_file.exists():
        logger.warning("Documents file not found. Please run data generation first.")
        return []
    
    with open(documents_file, 'r') as f:
        documents = json.load(f)
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def create_retriever(weights: Dict[str, float] = None) -> EnsembleRetriever:
    """Create and initialize an ensemble retriever."""
    retriever = EnsembleRetriever(weights)
    
    # Load documents
    documents = load_documents()
    if documents:
        retriever.add_documents(documents)
    
    return retriever

if __name__ == "__main__":
    # Test the retrieval system
    retriever = create_retriever()
    
    test_queries = [
        "Who owns the customer_data table?",
        "What tables contain email addresses?",
        "Show me data lineage for financial tables",
        "Which tables are owned by the Engineering department?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = retriever.retrieve(query)
        print(f"Retrieved {len(result.documents)} documents")
        print(f"Top score: {result.scores[0] if result.scores else 0}")
        print(f"Retrieval time: {result.total_time:.3f}s") 
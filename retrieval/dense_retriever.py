"""Dense retriever using multilingual E5"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch
from tqdm import tqdm

from .retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retriever using multilingual E5 embeddings"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        """
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.passages = []
        self.index = None
        
        # E5 requires prefixes
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "
    
    def index_corpus(self, passages: List[Dict]):
        """
        Index a corpus of passages
        
        Args:
            passages: List of dicts with at least 'text' key
        """
        self.passages = passages
        
        # Prepare texts with prefix
        texts = [f"{self.passage_prefix}{p['text']}" for p in passages]
        
        # Encode in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing passages"):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
        self.index.add(embeddings.astype(np.float32))
        
        print(f"  Indexed {len(passages)} passages")
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k documents for query
        """
        if self.index is None:
            raise RuntimeError("No corpus indexed. Call index_corpus() first.")
        
        # Prepare query with prefix
        query_text = f"{self.query_prefix}{query}"
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.passages[idx].copy()
            doc['score'] = float(score)
            doc['retrieval_rank'] = len(results) + 1
            results.append(doc)
        
        return results
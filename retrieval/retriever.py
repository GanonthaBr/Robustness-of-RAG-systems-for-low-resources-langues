"""Base retriever interface"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k documents for query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of documents with at least 'text' and 'score' keys
        """
        pass
    
    @abstractmethod
    def index_corpus(self, passages: List[Dict]):
        """Index a corpus of passages"""
        pass
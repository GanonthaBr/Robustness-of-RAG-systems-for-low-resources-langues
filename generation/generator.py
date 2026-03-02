"""Base generator interface"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseGenerator(ABC):
    """Abstract base class for all generators"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict with at least 'text' and optionally 'confidence'
        """
        pass
    
    @abstractmethod
    def get_confidence(self, text: str, prompt: str) -> float:
        """Get confidence score for generated text"""
        pass
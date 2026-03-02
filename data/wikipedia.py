"""Wikipedia corpus loader for African languages"""

from datasets import load_dataset
from typing import List, Dict, Optional
import os
import pickle


class WikipediaCorpus:
    """Load and manage Wikipedia passages for a language"""
    
    def __init__(self, language: str, cache_dir: Optional[str] = "./cache"):
        """
        Args:
            language: Language code ('swa', 'yor', 'kin', 'en')
            cache_dir: Directory to cache processed passages
        """
        self.language = language
        self.cache_dir = cache_dir
        self.passages = []
        
        # Language to dataset mapping
        self.wiki_configs = {
            'en': ('wikimedia/wikipedia', '20231101.en'),
            'swa': ('wikimedia/wikipedia', '20231101.sw'),
            'yor': ('wikimedia/wikipedia', '20231101.yo'),
            'kin': ('wikimedia/wikipedia', '20231101.rw'),
        }
        
        if language not in self.wiki_configs:
            raise ValueError(f"Language {language} not supported")
        
        self._load()
    
    def _load(self):
        """Load Wikipedia passages"""
        cache_file = f"{self.cache_dir}/wiki_{self.language}.pkl"
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading cached Wikipedia for {self.language}")
            with open(cache_file, 'rb') as f:
                self.passages = pickle.load(f)
            return
        
        # Load from HuggingFace
        print(f"Loading Wikipedia for {self.language} (first time, this may take a while)")
        dataset_name, config = self.wiki_configs[self.language]
        
        # Use streaming to avoid downloading full dataset
        dataset = load_dataset(dataset_name, config, split='train', streaming=True)
        
        # Take first N passages (adjust N based on your needs)
        max_passages = 10000  # Start with 10k, increase later
        for i, item in enumerate(dataset):
            if i >= max_passages:
                break
            
            # Clean and truncate text
            text = item['text'][:2000]  # Limit length
            if len(text.strip()) > 100:  # Only keep substantial passages
                self.passages.append({
                    'id': item.get('id', str(i)),
                    'title': item.get('title', ''),
                    'text': text,
                    'language': self.language
                })
        
        # Cache for next time
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(self.passages, f)
        
        print(f" Loaded {len(self.passages)} passages")
    
    def get_passages(self) -> List[Dict]:
        """Return all passages"""
        return self.passages
    
    def get_sample_passages(self, n: int = 5) -> List[Dict]:
        """Return n sample passages (for testing)"""
        import random
        return random.sample(self.passages, min(n, len(self.passages)))
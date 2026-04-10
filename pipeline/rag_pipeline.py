"""Main RAG pipeline orchestrator"""

from typing import Dict, List, Optional

from data.dataset import AfriQALoader
from data.wikipedia import WikipediaCorpus
from retrieval.dense_retriever import DenseRetriever
from generation.afrique_qwen import AfriqueQwenGenerator
from generation.prompts import PromptManager
from config.settings import RETRIEVAL_K, MAX_NEW_TOKENS, TEMPERATURE


class RAGPipeline:
    """Orchestrates the entire RAG pipeline"""
    
    def __init__(self, language: str, use_retrieval: bool = True, retriever=None, embedding_model: str = None):
        """
        Args:
            language: Target language
            use_retrieval: Whether to use retrieval
            retriever: Optional pre-built DenseRetriever (skips corpus load + indexing)
            embedding_model: Embedding model to use (e.g., 'intfloat/multilingual-e5-base')
        """
        self.language = language
        self.use_retrieval = use_retrieval
        
        print(f"\nInitializing RAG pipeline for {language}")
        
        # Initialize components
        self.prompt_manager = PromptManager(language)
        self.generator = AfriqueQwenGenerator()
        
        if use_retrieval:
            if retriever is not None:
                # Reuse the already-indexed retriever — skip corpus reload
                self.retriever = retriever
                print("  Retriever: using pre-built index")
            else:
                # Build from scratch
                self.corpus = WikipediaCorpus(language)
                if embedding_model:
                    self.retriever = DenseRetriever(model_name=embedding_model)
                else:
                    self.retriever = DenseRetriever()
                self.retriever.index_corpus(self.corpus.get_passages())
    
    def run(self, question: str, k: int = RETRIEVAL_K, 
            return_docs: bool = False) -> Dict:
        """
        Run RAG pipeline on a single question
        
        Args:
            question: Question in target language
            k: Number of documents to retrieve
            return_docs: Whether to return retrieved documents
            
        Returns:
            Dict with answer, confidence, and optionally documents
        """
        # Retrieve documents if enabled
        documents = []
        if self.use_retrieval:
            documents = self.retriever.retrieve(question, k=k)
        
        # Create prompt
        prompt = self.prompt_manager.create_prompt(
            question=question,
            documents=documents
        )
        
        # Generate answer
        result = self.generator.generate(
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )
        
        # Prepare output
        output = {
            'answer': result['text'],
            'confidence': result['confidence'],
            'prompt': prompt
        }
        
        if return_docs:
            output['documents'] = documents
        
        return output
    
    def run_batch(self, questions: List[str], **kwargs) -> List[Dict]:
        """Run pipeline on multiple questions"""
        results = []
        for q in questions:
            results.append(self.run(q, **kwargs))
        return results
"""Prompt management for different languages"""

from typing import Optional
from config.settings import PROMPT_TEMPLATES


class PromptManager:
    """Manage language-specific prompts"""
    
    def __init__(self, language: str):
        """
        Args:
            language: Language code ('en', 'swa', 'yor', 'kin')
        """
        self.language = language
        self.templates = PROMPT_TEMPLATES
    
    def create_prompt(self, question: str, documents: list, 
                      include_docs: bool = True) -> str:
        """
        Create a prompt for RAG
        
        Args:
            question: The question in the target language
            documents: List of retrieved documents
            include_docs: Whether to include documents (for non-RAG baseline)
        """
        if not include_docs:
            # Simple prompt without documents
            simple_prompts = {
                'en': f"Question: {question}\nAnswer:",
                'swa': f"Swali: {question}\nJibu:",
                'yor': f"Ìbéèrè: {question}\nDáhùn:",
                'kin': f"Ikibazo: {question}\nIgisubizo:",
            }
            return simple_prompts.get(self.language, simple_prompts['en'])
        
        # Format documents
        context = "\n\n".join([
            f"[Document {i+1}]: {doc['text']}" 
            for i, doc in enumerate(documents)
        ])
        
        # Get template for language
        template = self.templates.get(self.language, self.templates['en'])
        
        return template.format(context=context, question=question)
    
    def get_stop_tokens(self) -> list:
        """Get language-specific stop tokens"""
        # Common stop tokens across languages
        return ["\n\n", "Question:", "Swali:", "Ìbéèrè:", "Ikibazo:"]
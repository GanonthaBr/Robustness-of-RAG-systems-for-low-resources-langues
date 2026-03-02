"""AfriqueQwen generator implementation"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
import numpy as np

from .generator import BaseGenerator


class AfriqueQwenGenerator(BaseGenerator):
    """Generator using AfriqueQwen models"""
    
    def __init__(self, model_name: str = "McGill-NLP/AfriqueQwen-8B"):
        """
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        
        # Authenticate with HF Hub if token is available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and hf_token != "your_token_here":
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )
        
        self.device = next(self.model.parameters()).device
        print(f"   Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 0.7, return_confidence: bool = True) -> Dict:
        """
        Generate text from prompt
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate confidence (average token probability)
        confidence = None
        if return_confidence and hasattr(outputs, 'scores'):
            probs = []
            for score in outputs.scores:
                # Get probability of the chosen token
                token_probs = torch.softmax(score[0], dim=-1)
                chosen_token = outputs.sequences[0][inputs['input_ids'].shape[1] + len(probs)]
                probs.append(token_probs[chosen_token].item())
            confidence = np.mean(probs) if probs else None
        
        return {
            'text': generated_text,
            'confidence': confidence,
            'prompt': prompt
        }
    
    def get_confidence(self, text: str, prompt: str) -> float:
        """
        Get confidence score for a specific generated text
        (Simplified version - for production you'd compute log probs)
        """
        # This is a simplified version
        # For proper confidence, you'd need to compute log probabilities
        return 0.5  # Placeholder
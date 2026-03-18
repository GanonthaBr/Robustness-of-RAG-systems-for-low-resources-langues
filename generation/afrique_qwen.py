"""AfriqueQwen generator implementation"""

import os
import importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional
import numpy as np

from .generator import BaseGenerator


class AfriqueQwenGenerator(BaseGenerator):
    """Generator using AfriqueQwen models"""
    
    def __init__(self, model_name: str = "McGill-NLP/AfriqueQwen-8B", quantize: bool = True):
        """
        Args:
            model_name: HuggingFace model name
            quantize: Use 4-bit quantization (required for low VRAM GPUs)
        """
        self.model_name = model_name
        
        # Authenticate with HF Hub if token is available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and hf_token != "your_token_here":
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Build model loading kwargs
        load_kwargs = {"token": hf_token}
        
        if torch.cuda.is_available():
            if quantize:
                # 4-bit quantization needs bitsandbytes on CUDA systems.
                if importlib.util.find_spec("bitsandbytes") is not None:
                    print("  Using 4-bit quantization (NF4)")
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    print("  bitsandbytes not found; falling back to float16 loading")
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32
        
        load_kwargs["trust_remote_code"] = True
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except ValueError as exc:
            if "model type `qwen3`" in str(exc):
                raise RuntimeError(
                    "Your current Python/Transformers stack does not support Qwen3. "
                    "Use Python 3.10+ and upgrade dependencies with: "
                    "pip install -U 'transformers>=4.51.0' 'accelerate>=0.30.0'"
                ) from exc
            raise
        
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
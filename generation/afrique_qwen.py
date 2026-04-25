"""AfriqueQwen generator implementation with GPU resource management."""

import gc
import importlib.util
import os
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .generator import BaseGenerator


class AfriqueQwenGenerator(BaseGenerator):
    """Generator using AfriqueQwen or Qwen models with 4-bit quantization."""

    def __init__(
        self,
        model_name: str = "McGill-NLP/AfriqueQwen-8B",
        quantize: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name or local path.
            quantize:   Use 4-bit NF4 quantization (recommended for VRAM < 24 GB).
        """
        self.model_name = model_name

        # HuggingFace authentication
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict = {"token": hf_token, "trust_remote_code": True}

        if torch.cuda.is_available():
            if quantize and importlib.util.find_spec("bitsandbytes") is not None:
                print("  Using 4-bit quantization (NF4)")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,   # saves ~0.4 GB extra
                )
            else:
                if quantize:
                    print("  bitsandbytes not found; falling back to bfloat16")
                load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            print("  No CUDA detected; loading in float32 on CPU")
            load_kwargs["torch_dtype"] = torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except ValueError as exc:
            if "model type `qwen3`" in str(exc):
                raise RuntimeError(
                    "Your Transformers version does not support Qwen3. "
                    "Upgrade with: pip install -U 'transformers>=4.51.0' 'accelerate>=0.30.0'"
                ) from exc
            raise

        self.model.eval()  # disable dropout; slightly reduces memory during inference
        self.device = next(self.model.parameters()).device
        print(f"  Model loaded on {self.device}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"  VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        return_confidence: bool = False,   # default OFF to save VRAM
    ) -> Dict:
        """
        Generate text from a prompt.

        Args:
            prompt:            Input text.
            max_new_tokens:    Maximum tokens to generate.
            temperature:       Sampling temperature.
            return_confidence: Compute mean token probability.
                               Set to False during large sweeps to save VRAM.

        Returns:
            dict with keys: text, confidence (None if disabled), prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # Only materialise score tensors when confidence is requested.
                return_dict_in_generate=return_confidence,
                output_scores=return_confidence,
            )

        input_len = inputs["input_ids"].shape[1]

        if return_confidence:
            generated_ids = outputs.sequences[0][input_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            probs = []
            for step_idx, score in enumerate(outputs.scores):
                token_probs = torch.softmax(score[0], dim=-1)
                chosen_token = outputs.sequences[0][input_len + step_idx]
                probs.append(token_probs[chosen_token].item())
            confidence = float(np.mean(probs)) if probs else None

            # Explicitly free score tensors — they can be large for long outputs.
            del outputs
        else:
            # outputs is a plain tensor when return_dict_in_generate=False.
            generated_ids = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            confidence = None
            del outputs

        # Free input tensors and GPU cache.
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "text": generated_text,
            "confidence": confidence,
            "prompt": prompt,
        }

    # ------------------------------------------------------------------
    # Confidence helper (kept for compatibility)
    # ------------------------------------------------------------------

    def get_confidence(self, text: str, prompt: str) -> float:
        """
        Placeholder confidence scorer.
        For calibrated confidence use return_confidence=True in generate().
        """
        return 0.5

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        """Best-effort GPU cleanup when the object is garbage collected."""
        try:
            del self.model
            del self.tokenizer
        except AttributeError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
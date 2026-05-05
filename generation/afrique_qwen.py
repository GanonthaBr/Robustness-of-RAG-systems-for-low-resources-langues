"""AfriqueQwen generator implementation with GPU resource management and batch inference."""

import gc
import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .generator import BaseGenerator


class AfriqueQwenGenerator(BaseGenerator):
    """Generator using AfriqueQwen or Qwen models with 4-bit quantization and batch inference."""

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

        # Left-padding is required for batch generation with decoder-only models.
        # Right-padding causes the model to attend to padding tokens during generation.
        self.tokenizer.padding_side = "left"

        load_kwargs: Dict = {"token": hf_token, "trust_remote_code": True}

        if torch.cuda.is_available():
            if quantize and importlib.util.find_spec("bitsandbytes") is not None:
                print("  Using 4-bit quantization (NF4)")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
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

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"  Model loaded on {self.device}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"  VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")

    # ------------------------------------------------------------------
    # Single-prompt inference (kept for compatibility)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        return_confidence: bool = False,
        stop_strings: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate text from a single prompt.
        Internally calls generate_batch() with batch_size=1.
        For sweeps, call generate_batch() directly for GPU efficiency.
        """
        results = self.generate_batch(
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=1,
            return_confidence=return_confidence,
            stop_strings=stop_strings,
        )
        result = results[0]
        result["prompt"] = prompt
        return result

    # ------------------------------------------------------------------
    # Batch inference — primary method for robustness sweeps
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_at_stop(text: str, stop_strings: List[str]) -> str:
        """Truncate generated text at the first occurrence of any stop string."""
        for stop in stop_strings:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        batch_size: int = 8,
        return_confidence: bool = False,
        stop_strings: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Generate text for a list of prompts in batches.
        Significantly more GPU-efficient than calling generate() one at a time.

        Args:
            prompts:           List of input prompts.
            max_new_tokens:    Maximum tokens to generate per prompt.
            temperature:       Sampling temperature.
            batch_size:        Prompts per GPU forward pass.
                               - Start with 8 (safe for 4-bit 7-8B on 48GB)
                               - Reduce to 4 if you get OOM errors
                               - Increase to 16 if VRAM headroom allows
            return_confidence: Compute mean token probability per prompt.
                               Leave False during large sweeps to save VRAM.

        Returns:
            List of dicts with keys: text, confidence.
        """
        all_results = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start: batch_start + batch_size]

            # Tokenize with left-padding so all sequences end at the same
            # position — this is critical for decoder-only batch generation.
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=return_confidence,
                    output_scores=return_confidence,
                )

            if return_confidence:
                sequences = outputs.sequences
                scores = outputs.scores
            else:
                sequences = outputs
                scores = None

            for seq_idx, seq in enumerate(sequences):
                generated_ids = seq[input_len:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if stop_strings:
                    text = self._truncate_at_stop(text, stop_strings)

                confidence = None
                if return_confidence and scores is not None:
                    probs = []
                    for step_idx, score in enumerate(scores):
                        token_probs = torch.softmax(score[seq_idx], dim=-1)
                        chosen_token = seq[input_len + step_idx]
                        probs.append(token_probs[chosen_token].item())
                    confidence = float(np.mean(probs)) if probs else None

                all_results.append({"text": text, "confidence": confidence})

            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_results

    # ------------------------------------------------------------------
    # Confidence helper (kept for compatibility)
    # ------------------------------------------------------------------

    def get_confidence(self, text: str, prompt: str) -> float:
        """
        Placeholder confidence scorer.
        For calibrated confidence use return_confidence=True in generate_batch().
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
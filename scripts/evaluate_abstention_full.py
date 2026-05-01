#!/usr/bin/env python3
"""Full abstention analysis: generate predictions, evaluate heuristics, plot curves."""

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if sys.version_info[0] < 3:
    raise RuntimeError("evaluate_abstention_full.py requires Python 3.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from config.settings import EMBEDDING_MODELS, LLM_MODELS
from data.dataset import AfriQALoader
from data.wikipedia import WikipediaCorpus
from evaluation.metrics import Evaluator
from generation.afrique_qwen import AfriqueQwenGenerator
from generation.prompts import PromptManager
from retrieval.dense_retriever import DenseRetriever
from utils.helpers import load_json, save_json

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

# Best-k from clean comparison runs
BEST_K_BY_MODEL = {
    "afriqueqwen-8b": {"swa": 10, "yor": 3, "kin": 10},
    "qwen2.5-7b-instruct": {"swa": 3, "yor": 3, "kin": 10},
}


def _model_tag(model_key):
    return str(model_key).replace("/", "-").replace(":", "-")


def _resolve_model_name(model_key):
    if model_key in LLM_MODELS:
        return LLM_MODELS[model_key]
    if model_key:
        return model_key
    return "afriqueqwen-8b"


def _sample_examples(examples, num_examples, seed):
    if num_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), num_examples)
    return [examples[i] for i in sorted(indices)]


def _build_golds(examples):
    golds_local = []
    golds_en = []
    for ex in examples:
        answers = ex.get("answers", "")
        if isinstance(answers, list):
            golds_local.append(answers[0] if answers else "")
        else:
            golds_local.append(answers)
        golds_en.append(ex.get("translated_answer", ""))
    return golds_local, golds_en


def _get_confidence_from_generation(result):
    """Extract confidence score from generation result."""
    return result.get("confidence", 0.0)


def _run_inference(generator, prompt_manager, examples, retrieved_docs, language, evaluator):
    """Run inference and collect predictions with confidence scores."""
    predictions = []
    confidences = []
    golds_local = []
    golds_en = []

    for i, ex in enumerate(examples):
        prompt = prompt_manager.create_prompt(
            question=ex["question"],
            documents=retrieved_docs[i],
            include_docs=True,
        )

        result = generator.generate(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.7,
            return_confidence=True,
        )

        predictions.append(result.get("text", ""))
        confidences.append(_get_confidence_from_generation(result))

        answers = ex.get("answers", "")
        if isinstance(answers, list):
            golds_local.append(answers[0] if answers else "")
        else:
            golds_local.append(answers)
        golds_en.append(ex.get("translated_answer", ""))

    metrics = evaluator.evaluate_batch(predictions, golds_local, golds_en)

    return {
        "predictions": predictions,
        "confidences": confidences,
        "golds_local": golds_local,
        "golds_en": golds_en,
        "metrics": metrics,
    }


def _compute_abstention_curves(predictions, confidences, golds_local, evaluator):
    """
    Compute Accuracy-Rejection Curves by sweeping confidence thresholds.
    Returns metrics at each threshold.
    """
    confidences_np = np.array(confidences)
    thresholds = np.linspace(0.0, 1.0, 21)  # 0, 0.05, 0.1, ..., 1.0

    curves = []

    for threshold in thresholds:
        # Identify which predictions to keep (confidence >= threshold)
        keep_mask = confidences_np >= threshold
        num_kept = np.sum(keep_mask)
        rejection_rate = 1.0 - (num_kept / len(predictions))

        if num_kept == 0:
            # All examples rejected
            curves.append(
                {
                    "threshold": float(threshold),
                    "rejection_rate": float(rejection_rate),
                    "accuracy": 0.0,
                    "num_answered": 0,
                    "num_correct": 0,
                }
            )
        else:
            # Evaluate on kept examples
            kept_preds = [p for i, p in enumerate(predictions) if keep_mask[i]]
            kept_golds = [g for i, g in enumerate(golds_local) if keep_mask[i]]

            # Simple exact match
            num_correct = sum(
                1 for p, g in zip(kept_preds, kept_golds) if p.strip().lower() == g.strip().lower()
            )
            accuracy = num_correct / num_kept

            curves.append(
                {
                    "threshold": float(threshold),
                    "rejection_rate": float(rejection_rate),
                    "accuracy": float(accuracy),
                    "num_answered": int(num_kept),
                    "num_correct": int(num_correct),
                }
            )

    return curves


def main(
    num_examples,
    seeds,
    languages,
    llm_models,
    embedding_model,
    output_dir,
):
    """Run full abstention evaluation."""
    from config.settings import MAX_NEW_TOKENS, TEMPERATURE

    max_new_tokens = MAX_NEW_TOKENS
    temperature = TEMPERATURE

    loader = AfriQALoader()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("FULL ABSTENTION ANALYSIS")
    print("=" * 96)
    print(f"Models       : {llm_models}")
    print(f"Languages    : {languages}")
    print(f"Seeds        : {seeds}")
    print(f"Examples/lang: {num_examples}")
    print(f"Embedding    : {embedding_model}")
    print(f"Output dir   : {output_root}")

    # Build retrievers
    retriever_by_language = {}
    passage_pool_by_language = {}
    for language in languages:
        print(f"\nPreparing retriever for {language}...")
        corpus = WikipediaCorpus(language)
        passage_pool = corpus.get_passages()
        retriever = DenseRetriever(model_name=embedding_model)
        retriever.index_corpus(passage_pool)
        retriever_by_language[language] = retriever
        passage_pool_by_language[language] = passage_pool

    all_results = {}

    for llm_key in llm_models:
        model_name = _resolve_model_name(llm_key)
        llm_tag = _model_tag(llm_key)

        print("\n" + "#" * 96)
        print(f"MODEL: {llm_key} -> {model_name}")
        print("#" * 96)

        generator = AfriqueQwenGenerator(model_name=model_name)
        all_results[llm_key] = {"by_language": {}}

        for language in languages:
            print(f"\nLanguage: {language}")
            prompt_manager = PromptManager(language)
            evaluator = Evaluator(language=language)
            retriever = retriever_by_language[language]
            best_k = BEST_K_BY_MODEL.get(llm_key, {}).get(language, 5)

            all_examples = loader.load(language, split="test", num_samples=None)
            all_results[llm_key]["by_language"][language] = {"by_seed": {}}

            for seed in seeds:
                print(f"  Seed {seed}...")
                examples = _sample_examples(all_examples, num_examples, seed)
                golds_local, golds_en = _build_golds(examples)

                # Retrieve docs
                retrieved_docs = []
                for ex in examples:
                    retrieved_docs.append(retriever.retrieve(ex["question"], k=int(best_k)))

                # Run inference
                inference_results = _run_inference(
                    generator, prompt_manager, examples, retrieved_docs, language, evaluator
                )

                # Compute abstention curves
                curves = _compute_abstention_curves(
                    inference_results["predictions"],
                    inference_results["confidences"],
                    inference_results["golds_local"],
                    evaluator,
                )

                all_results[llm_key]["by_language"][language]["by_seed"][seed] = {
                    "num_examples": len(examples),
                    "num_kept_at_threshold_0": curves[0]["num_answered"],
                    "accuracy_at_threshold_0": curves[0]["accuracy"],
                    "curves": curves,
                    "predictions": inference_results["predictions"],
                    "confidences": inference_results["confidences"],
                }

            # Save per-language results
            lang_output = output_root / f"abstention_{llm_tag}_{language}.json"
            save_json(all_results[llm_key]["by_language"][language], str(lang_output))
            print(f"  Saved: {lang_output}")

        del generator

    # Save complete results
    output_file = output_root / "abstention_full_analysis.json"
    save_json(all_results, str(output_file))
    print(f"\nSaved full analysis: {output_file}")

    print("\nAbstention analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full abstention analysis: inference + evaluation + curves"
    )
    parser.add_argument(
        "--num-examples", type=int, default=50, help="Examples per language/seed"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds to evaluate",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["swa", "yor", "kin"],
        help="Languages to evaluate",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        choices=list(LLM_MODELS.keys()),
        help="Single LLM model",
    )
    parser.add_argument(
        "--all-llms",
        action="store_true",
        help="Run all models",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="e5-base",
        help="Embedding model key",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    if args.all_llms:
        model_keys = list(LLM_MODELS.keys())
    elif args.llm_model:
        model_keys = [args.llm_model]
    else:
        model_keys = ["afriqueqwen-8b"]

    emb_model_path = EMBEDDING_MODELS.get(args.embedding_model, EMBEDDING_MODELS["e5-base"])
    output_dir = args.output_dir or str(PROJECT_ROOT / "results/abstention_analysis")

    main(
        num_examples=args.num_examples,
        seeds=args.seeds,
        languages=args.languages,
        llm_models=model_keys,
        embedding_model=emb_model_path,
        output_dir=output_dir,
    )

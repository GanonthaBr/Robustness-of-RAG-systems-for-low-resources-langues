#!/usr/bin/env python3
"""Run RAG vs LLM-only in one pass on the same sampled examples."""

import argparse
import os
import random
import sys

if sys.version_info[0] < 3:
    raise RuntimeError(
        "run_rag_vs_llm_once.py requires Python 3. "
        "Run with 'python3 scripts/run_rag_vs_llm_once.py --num-examples 100'."
    )

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from config.settings import EMBEDDING_MODELS, LLM_MODELS, RETRIEVAL_K
from data.dataset import AfriQALoader
from evaluation.metrics import Evaluator, RetrieverEvaluator
from pipeline.rag_pipeline import RAGPipeline
from utils.helpers import save_json

load_dotenv(PROJECT_ROOT / ".env")


def _sample_examples(examples, num_examples, seed):
    """Deterministically sample examples so RAG and LLM-only share the same items."""
    if num_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), num_examples)
    return [examples[i] for i in indices]


def _run_mode(pipeline, examples, evaluator, capture_docs, retrieval_k=None):
    """Run a single mode (RAG or LLM-only) and return predictions + docs + eval metrics."""
    predictions = []
    golds_local = []
    golds_en = []
    all_retrieved_docs = []

    for i, ex in enumerate(examples):
        print("  Example {}/{}".format(i + 1, len(examples)))
        if retrieval_k is None:
            result = pipeline.run(ex["question"], return_docs=capture_docs)
        else:
            result = pipeline.run(ex["question"], k=int(retrieval_k), return_docs=capture_docs)
        predictions.append(result["answer"])

        answers = ex.get("answers", "")
        if isinstance(answers, list):
            golds_local.append(answers[0] if answers else "")
        else:
            golds_local.append(answers)

        golds_en.append(ex.get("translated_answer", ""))

        docs = result.get("documents", []) if capture_docs else []
        all_retrieved_docs.append(docs)

    generation_metrics = evaluator.evaluate_batch(predictions, golds_local, golds_en)

    return {
        "generation_metrics": generation_metrics,
        "retrieved_docs": all_retrieved_docs,
    }


def main(num_examples, seed, output_file=None, k_by_language=None, llm_model='afriqueqwen-8b'):
    """Run RAG vs LLM-only comparison using fixed E5 retriever for RAG mode."""
    languages = ["swa", "yor", "kin"]
    e5_model = EMBEDDING_MODELS.get("e5-base", "intfloat/multilingual-e5-base")

    print("=" * 80)
    print("RAG VS LLM-ONLY (SAME SAMPLES)")
    print("=" * 80)
    print("Languages: {}".format(languages))
    print("Examples per language: {}".format(num_examples))
    print("Sampling seed: {}".format(seed))
    print("RAG embedding: e5-base ({})".format(e5_model))
    print("LLM model: {}".format(llm_model))
    if k_by_language:
        print("Language-specific k: {}".format(k_by_language))
    else:
        print("Global k: {}".format(RETRIEVAL_K))

    loader = AfriQALoader()
    results = {
        "metadata": {
            "languages": languages,
            "num_examples": num_examples,
            "seed": seed,
            "rag_embedding": "e5-base",
            "rag_embedding_model": e5_model,
            "llm_model": llm_model,
            "k_by_language": k_by_language or {lang: RETRIEVAL_K for lang in languages},
        },
        "by_language": {},
    }

    for language in languages:
        print("\n" + "-" * 80)
        print("Language: {}".format(language))
        print("-" * 80)

        all_examples = loader.load(language, split="test", num_samples=None)
        examples = _sample_examples(all_examples, num_examples, seed)

        # RAG run (with E5 retrieval)
        print("Running RAG mode...")
        rag_pipeline = RAGPipeline(language, use_retrieval=True, embedding_model=e5_model, llm_model=llm_model)
        rag_evaluator = Evaluator(language=language)
        rag_k = int((k_by_language or {}).get(language, RETRIEVAL_K))
        rag_out = _run_mode(rag_pipeline, examples, rag_evaluator, retrieval_k=rag_k, capture_docs=True)
        rag_retriever_metrics = RetrieverEvaluator.evaluate_retrieval(rag_out["retrieved_docs"])

        # LLM-only run (no retrieval)
        print("Running LLM-only mode...")
        llm_pipeline = RAGPipeline(language, use_retrieval=False, llm_model=llm_model)
        llm_evaluator = Evaluator(language=language)
        llm_out = _run_mode(llm_pipeline, examples, llm_evaluator, capture_docs=False)

        rag_gen = rag_out["generation_metrics"]
        llm_gen = llm_out["generation_metrics"]

        delta = {
            "correct_rate": rag_gen.get("correct_rate", 0.0) - llm_gen.get("correct_rate", 0.0),
            "contains_gold_local": rag_gen.get("contains_gold_local", 0.0) - llm_gen.get("contains_gold_local", 0.0),
            "contains_gold_english": rag_gen.get("contains_gold_english", 0.0) - llm_gen.get("contains_gold_english", 0.0),
            "abstention_rate": rag_gen.get("abstention_rate", 0.0) - llm_gen.get("abstention_rate", 0.0),
        }

        results["by_language"][language] = {
            "rag": {
                "retriever_metrics": rag_retriever_metrics,
                "generation_metrics": rag_gen,
            },
            "llm_only": {
                "retriever_metrics": {
                    "mean_similarity": 0.0,
                    "mean_top1_similarity": 0.0,
                    "mean_top5_similarity": 0.0,
                    "max_similarity": 0.0,
                    "min_similarity": 0.0,
                    "num_queries": len(examples),
                },
                "generation_metrics": llm_gen,
            },
            "delta_rag_minus_llm_only": delta,
        }

        print("\nSummary {}:".format(language.upper()))
        print(
            "  RAG      correct={:.1%}, local={:.1%}, en={:.1%}, abstain={:.1%}".format(
                rag_gen.get("correct_rate", 0.0),
                rag_gen.get("contains_gold_local", 0.0),
                rag_gen.get("contains_gold_english", 0.0),
                rag_gen.get("abstention_rate", 0.0),
            )
        )
        print(
            "  LLM-only correct={:.1%}, local={:.1%}, en={:.1%}, abstain={:.1%}".format(
                llm_gen.get("correct_rate", 0.0),
                llm_gen.get("contains_gold_local", 0.0),
                llm_gen.get("contains_gold_english", 0.0),
                llm_gen.get("abstention_rate", 0.0),
            )
        )
        print(
            "  Delta    correct={:+.1%}, local={:+.1%}, en={:+.1%}, abstain={:+.1%}".format(
                delta["correct_rate"],
                delta["contains_gold_local"],
                delta["contains_gold_english"],
                delta["abstention_rate"],
            )
        )

    llm_tag = str(llm_model).replace("/", "-").replace(":", "-")

    if output_file:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = PROJECT_ROOT / "results/rag_vs_llm_only_e5_100_{}.json".format(llm_tag)

    save_json(results, str(output_path))
    print("\nSaved combined results: {}".format(output_path))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG and LLM-only together on same samples")
    parser.add_argument("--num-examples", type=int, default=100, help="Examples per language")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--output-file", type=str, default=None, help="Optional output JSON path")
    parser.add_argument(
        "--llm-model",
        type=str,
        choices=list(LLM_MODELS.keys()),
        default='afriqueqwen-8b',
        help="LLM model key from config.LLM_MODELS",
    )
    args = parser.parse_args()

    main(num_examples=args.num_examples, seed=args.seed, output_file=args.output_file, llm_model=args.llm_model)

#!/usr/bin/env python
"""Compare saved RAG and LLM-only summaries."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from utils.helpers import load_json, save_json


def safe_get(metrics_block, key):
    return float(metrics_block.get(key, 0.0)) if metrics_block else 0.0


def main():
    rag_path = PROJECT_ROOT / "results/all_languages_enhanced_summary.json"
    llm_only_path = PROJECT_ROOT / "results/all_languages_llm_only_summary.json"

    if not rag_path.exists():
        raise FileNotFoundError(f"Missing RAG summary: {rag_path}")
    if not llm_only_path.exists():
        raise FileNotFoundError(f"Missing LLM-only summary: {llm_only_path}")

    rag = load_json(str(rag_path))
    llm_only = load_json(str(llm_only_path))

    languages = sorted(set(rag.keys()).intersection(set(llm_only.keys())))
    comparison = {}

    print("=" * 84)
    print("RAG VS LLM-ONLY COMPARISON")
    print("=" * 84)
    print(f"{'Language':<10} {'RAG Correct':<12} {'LLM Correct':<12} {'Delta (RAG-LLM)':<16}")
    print("-" * 84)

    for language in languages:
        rag_gen = rag[language].get("generation_metrics", rag[language])
        llm_gen = llm_only[language].get("generation_metrics", llm_only[language])

        rag_correct = safe_get(rag_gen, "correct_rate")
        llm_correct = safe_get(llm_gen, "correct_rate")
        delta = rag_correct - llm_correct

        comparison[language] = {
            "rag_correct_rate": rag_correct,
            "llm_only_correct_rate": llm_correct,
            "delta_correct_rate": delta,
            "rag_contains_gold_local": safe_get(rag_gen, "contains_gold_local"),
            "llm_only_contains_gold_local": safe_get(llm_gen, "contains_gold_local"),
            "rag_contains_gold_english": safe_get(rag_gen, "contains_gold_english"),
            "llm_only_contains_gold_english": safe_get(llm_gen, "contains_gold_english"),
            "rag_abstention_rate": safe_get(rag_gen, "abstention_rate"),
            "llm_only_abstention_rate": safe_get(llm_gen, "abstention_rate"),
        }

        print(f"{language:<10} {rag_correct:<12.1%} {llm_correct:<12.1%} {delta:<16.1%}")

    save_json(comparison, str(PROJECT_ROOT / "results/rag_vs_llm_only_comparison.json"))
    print("\nSaved: results/rag_vs_llm_only_comparison.json")


if __name__ == "__main__":
    main()

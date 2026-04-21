# Robustness of RAG systems for Low Resource Languages

## WorkFlow

The project workflow begins by selecting 200 test queries each for English, Swahili, Yoruba, and Kinyarwanda from AfriQA, along with a parallel set of multiple-choice queries for Swahili and Yoruba from IrokoBench's AfriMMLU subset to validate generalizability. For each query, we retrieve the top ten most relevant documents from the corresponding language's Wikipedia corpus using a multilingual embedding model, then systematically inject four noise types—irrelevant, contradictory, partially correct, and translation artifacts—at three severity levels of ten, thirty, and fifty percent corruption, creating thirteen distinct conditions per query. Each condition is passed to the AfriqueQwen-8B language model with prompts crafted in the query's language, generating either an answer or an abstention along with confidence scores derived from token log probabilities. Generated answers are translated to English and evaluated against gold labels using exact match and F1 scores, while confidence calibration is assessed through Expected Calibration Error and reliability diagrams. Finally, we evaluate three abstention heuristics—self-consistency, NLI entailment, and logit-based confidence—sweeping thresholds to generate Accuracy-Rejection Curves and identify language-specific optimal thresholds, producing a comprehensive analysis of how noise differentially affects African-language RAG and whether abstention mechanisms require calibration tailored to each language

## Env
source .venv310/bin/activate

## LLM Comparison Setup

You can compare multiple free/open Hugging Face LLMs by selecting a model key from `LLM_MODELS` in `config/settings.py` and passing it via script flags (for example `--llm-model qwen2.5-7b-instruct`).

To generate a direct LLM-vs-LLM comparison table after running `--all-llms` k-sweep:

```bash
python3 scripts/compare_llms_k_sweep.py --index-file results/rag_k_sweep_all_llms_multiseed_index.json
```

This produces:

- `results/llm_comparison_from_k_sweep.json`
- `results/llm_comparison_from_k_sweep.md`

## Robustness Full Run (Checkpoint + Progress)

Launch one resumable robustness run for all configured LLMs:

```bash
python3 scripts/run_robustness_multiseed.py --all-llms --num-examples 50 --seeds 42 43 44 --languages swa yor kin
```

Optional faster run (single model):

```bash
python3 scripts/run_robustness_multiseed.py --llm-model afriqueqwen-8b --num-examples 50 --seeds 42 43 44
```

What it writes:

- Per-condition checkpoints: `results/robustness_checkpoints/robustness_<model>_<lang>_seed<seed>_<condition>.json`
- Resume state file: `results/robustness_checkpoints/robustness_run_state.json`
- Per-model summary: `results/robustness_<model>_multiseed_summary.json`
- Global run index: `results/robustness_all_llms_index.json`

Monitoring during run:

- Console progress logs include percent complete, elapsed time, and ETA
- Resume works by default; use `--no-resume` to force rerun

Post-run processing (after robustness run finishes):

```bash
python3 scripts/postprocess_robustness_results.py --index-file results/robustness_all_llms_index.json
python3 scripts/build_final_paper_draft.py --clean-file results/llm_comparison_from_k_sweep.json --robustness-file results/robustness_postrun_summary.json
```

Post-run outputs:

- `results/robustness_postrun_summary.json`
- `results/robustness_postrun_long.csv`
- `results/robustness_postrun_tables.md`
- `results/final_paper_results_draft.md`
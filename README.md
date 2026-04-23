# DeepReview Evidence-Driven Scoring Framework

An evidence-driven automated scoring framework for academic paper review, validated on the DeepReview dataset.

## Overview

This project implements a complete experimental platform to validate:
1. Baseline performance (Fast/Standard/Best modes aligned with DeepReviewer)
2. Whether structured evidence extraction improves scoring quality
3. Whether hypergraph-enhanced retrieval is more effective than plain text
4. Whether multi-agent scoring and arbitration improve stability
5. Whether risk control mechanisms reduce erroneous auto-decisions
6. Whether templated explanations are more faithful than free-generation

## Project Structure

```
project/
├── data/
│   ├── raw/                  # Place DeepReview sample.json here
│   ├── processed/            # Adapted unified format data
│   └── splits/               # Train/dev/test splits
├── configs/
│   ├── deepreview.yaml       # DeepReview dataset config (dimensions, metrics)
│   └── llm.yaml              # LLM provider config (ollama / OpenAI)
├── src/
│   ├── adapters/             # Dataset adapters
│   ├── preprocess/           # LaTeX parsing
│   ├── evidence/             # Evidence extraction
│   ├── rubric/               # Rubric definitions
│   ├── graph/                # Hypergraph builder and retrieval
│   ├── agents/               # Multi-agent workflow
│   ├── scoring/              # Baseline and aggregation
│   ├── evaluation/           # Official metrics (MSE/MAE/Spearman/Decision F1/Pairwise)
│   ├── utils/                # LLM wrapper, parser, logger
│   └── cli/                  # CLI entry point
├── experiments/              # Experiment outputs
├── scripts/                  # Run scripts
├── tests/                    # Unit tests
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### LLM Provider

Edit `configs/llm.yaml` to set your LLM provider:

**Local (Ollama)**:
```yaml
default_provider: "ollama"
ollama:
  base_url: "http://localhost:11434"
  models:
    local_default:
      name: "qwen3.5:4b"
    cloud_default:
      name: "deepseek-r1:14b"
```

**Remote (OpenAI-compatible)**:
```yaml
default_provider: "openai"
openai:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
```

## Usage

### 1. Preprocess Data

```bash
python src/cli/main.py preprocess \
    --input "Research/evaluate/DeepReview/sample.json" \
    --output "data/processed/deepreview_processed.json"
```

### 2. Evaluate Official Predictions

Evaluate the built-in predictions (`pred_fast_mode`, `pred_standard_mode`, `pred_best_mode`) from the raw dataset:

```bash
python src/cli/main.py evaluate --official \
    --input "Research/evaluate/DeepReview/sample.json" \
    --output-dir "experiments/deepreview_baseline"
```

### 3. Run Baselines

**Baseline 1 - Fast Mode** (pure LLM end-to-end):
```bash
python scripts/run_baseline.py --mode fast --samples 10 --evaluate
```

**Baseline 2 - Standard Mode** (multi-reviewer simulation):
```bash
python scripts/run_baseline.py --mode standard --samples 10 --evaluate
```

**Baseline 3 - Best Mode** (retrieval-enhanced):
```bash
python scripts/run_baseline.py --mode best --samples 10 --evaluate
```

### 4. Run Our Enhanced Method

```bash
python scripts/run_ours.py --samples 10 --use-llm-evidence
```

### 5. Run Ablation Experiments

```bash
# Full system
python scripts/run_ablation.py --variant full --samples 10

# Without hypergraph
python scripts/run_ablation.py --variant no_hg --samples 10

# Without LLM evidence
python scripts/run_ablation.py --variant no_evidence --samples 10

# Without risk control
python scripts/run_ablation.py --variant no_risk --samples 10
```

### 6. Extract Evidence

```bash
python src/cli/main.py evidence \
    --input "data/processed/deepreview_processed.json" \
    --output "experiments/evidence_results.json"
```

## Evaluation Metrics

Aligned with official DeepReview implementation (`evalate.py`):

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error per dimension |
| MAE | Mean Absolute Error per dimension |
| Spearman | Spearman rank correlation |
| Decision Accuracy | Accept/Reject accuracy |
| Decision F1 (macro) | Macro-averaged F1 for decisions |
| Pairwise Accuracy | Pairwise ranking accuracy |

**Note**: RMSE, Pearson, and QWK are computed as extended metrics but are not part of the official evaluation.

## Alignment with Official DeepReview

This implementation strictly follows the official DeepReview code in `Research/`:

- **Output Format**: Compatible with `\boxed_review{}` and `\boxed_simreviewers{}`
- **Parser**: Aligns with `get_reviewer_score()` and `_parse_review()`
- **Metrics**: Replicates `evalate.py` exactly (MSE/MAE/Spearman/Decision F1/Pairwise Acc)
- **Prompts**: Follows Fast/Standard/Best mode system prompts from `deep_reviewer.py`

## Key Design Decisions

1. **Data Format**: DeepReview uses LaTeX paper text, not code blocks. The adapter maps real fields to a unified internal schema.
2. **Rubric**: DeepReview has fixed 4 dimensions (Rating 1-10, Soundness/Presentation/Contribution 1-5).
3. **Ground Truth**: Averaged across multiple human reviewers (aligns with official).
4. **Evidence Types**: Adapted for academic papers (claim/method/experiment/reference/consistency).
5. **LLM Wrapper**: Unified interface supporting both local Ollama and remote APIs.

## License

Research use only.

# NASA Science Repos Benchmark

A flexible benchmarking system for evaluating different text representations (README, cleaned README, README+topics, README+additional context) with various retrieval methods (BM25, embedding-based) on the NASA Science Repositories dataset.

## Overview

This benchmark helps compare retrieval performance across different text views to determine which information sources (raw README, cleaned README, topics, additional context) are most valuable for finding relevant NASA science repositories.

## Features

- **Multiple Text Views**: Test different combinations of repository metadata
  - `readme`: Raw README only
  - `readme_cleaned`: Cleaned/processed README
  - `readme_and_topics`: README + repository topics
  - `readme_and_additional_context`: README + enriched context
  - `enriched`: Full enriched view with all metadata
  - `full`: All fields combined

- **Multiple Retrieval Methods**:
  - BM25 (sparse retrieval)
  - Embedding-based (dense retrieval with sentence transformers)

- **Comprehensive Metrics**:
  - MRR@K (Mean Reciprocal Rank)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Recall@K
  - K values: 1, 5, 10

- **Division-Specific Evaluation**:
  - Earth Science
  - Astrophysics
  - Planetary
  - Holistic (overall)

## Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
uv sync
```

## Usage

### Quick Start - BM25 Benchmark with Division Breakdown

Run BM25 on all 4 text views with division-specific evaluation:

```bash
uv run python main.py --by-division --views readme readme_cleaned readme_and_topics readme_and_additional_context
```

This will:
- Run BM25 retrieval (default)
- Test all 4 text views
- Show results for Earth Science, Astrophysics, Planetary, and Holistic divisions
- Save to: `results/bm25_benchmark_results_by_division.csv`

### Division-Specific Benchmark Examples

```bash
# BM25 on readme only (default)
uv run python main.py --by-division

# BM25 on all 4 views
uv run python main.py --by-division --views readme readme_cleaned readme_and_topics readme_and_additional_context

# Embedding retrieval with division breakdown
uv run python main.py --by-division --retrievers embedding --embedding-model all-MiniLM-L6-v2 --views readme

# Run both BM25 and embedding
uv run python main.py --by-division --retrievers bm25 embedding --views readme
```

Output files:
- BM25: `results/bm25_benchmark_results_by_division.csv`
- Embedding: `results/embedding_{model_name}_benchmark_results_by_division.csv`

### Standard Benchmark (No Division Breakdown)

```bash
# Test only BM25 with README views
uv run python main.py --retrievers bm25 --views readme readme_cleaned

# Test only embedding retrieval
uv run python main.py --retrievers embedding --views readme_and_topics

# Use a different embedding model
uv run python main.py --retrievers embedding --embedding-model sentence-transformers/all-mpnet-base-v2

# Use NASA Science-tuned embedding model
uv run python main.py --retrievers embedding --embedding-model nasa-impact/nasa-smd-ibm-st-v2 --by-division

# Skip visualization plots
uv run python main.py --no-plots

# Save results to custom location
uv run python main.py --output my_results.csv
```

### Available Options

```
--views [VIEW ...]              Text views to benchmark (default: readme)
                                Options: readme, readme_cleaned, readme_and_topics, readme_and_additional_context
--retrievers [RET ...]          Retrieval methods (default: bm25)
                                Options: bm25, embedding
--embedding-model MODEL         Sentence transformer model (default: all-MiniLM-L6-v2)
--by-division                   Run with division-specific evaluation (Earth, Astro, Planetary, Holistic)
--output PATH                   Output CSV path (default: results/benchmark_results.csv)
--no-plots                      Skip creating visualization plots
```

## Testing Components

### Test Data Loading

```bash
uv run python -c "from src.data_loader import load_benchmark_corpus, load_parent_corpus, enrich_corpus; bc = load_benchmark_corpus(); pc = load_parent_corpus(); ec = enrich_corpus(bc, pc); print(f'Benchmark: {len(bc)}, Parent: {len(pc)}, Enriched: {len(ec)}')"
```

### Test View Generation

```bash
uv run python -c "from src.data_loader import load_benchmark_corpus, load_parent_corpus, enrich_corpus; from src.view_generator import ViewGenerator; bc = load_benchmark_corpus(); pc = load_parent_corpus(); corpus = enrich_corpus(bc, pc); vg = ViewGenerator(); view = vg.create_text_view(corpus, 'readme_and_topics'); print(f'Created view with {len(view)} documents')"
```

## Output

The benchmark produces:

1. **Console Output**: Formatted table with all metrics
2. **CSV Results**: Detailed results in `results/benchmark_results.csv`
3. **Visualization Plots** (in `results/plots/`):
   - `recall_at_*.png`: Bar charts comparing recall across views
   - `ndcg_at_*.png`: Bar charts comparing NDCG across views
   - `*_heatmap.png`: Heatmaps showing metric values

## Project Structure

```
code-search-experiments/
├── main.py                    # Entry point
├── pyproject.toml            # Dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load and enrich corpus
│   ├── view_generator.py     # Create text views
│   ├── metrics.py            # Recall, NDCG calculation
│   ├── benchmark.py          # Main benchmark runner
│   ├── visualize.py          # Results visualization
│   └── retrievers/
│       ├── __init__.py
│       ├── base_retriever.py
│       ├── bm25_retriever.py
│       └── embedding_retriever.py
├── data/                     # Local data cache
└── results/                  # Output directory
    ├── benchmark_results.csv
    └── plots/
```

## Dataset Information

- **Benchmark**: [nasa-impact/nasa-science-repos-sme-benchmark](https://huggingface.co/datasets/nasa-impact/nasa-science-repos-sme-benchmark)
  - 5,264 repositories (corpus) with 15 metadata columns
  - 219 queries
  - 259 relevance judgments (qrels)
  - Divisions: Earth Science (158 qrels), Astrophysics (61 qrels), Planetary (40 qrels)

- **Parent Dataset**: [nasa-impact/nasa-science-github-repos](https://huggingface.co/datasets/nasa-impact/nasa-science-github-repos)
  - 5,264 repositories with rich metadata
  - README, cleaned README, topics, additional context, etc.

## Example Output

### Division-Specific Benchmark Results

```
====================================================================================================
BM25 BENCHMARK RESULTS BY DIVISION
====================================================================================================

View: readme
----------------------------------------------------------------------------------------------------
Division        MRR@1   NDCG@1   MRR@5   NDCG@5   MRR@10  NDCG@10
----------------------------------------------------------------------------------------------------
earth           0.0129  0.0129  0.0151  0.0161  0.0167  0.0202
astro           0.2188  0.2188  0.2719  0.2251  0.2758  0.2321
planetary       0.0909  0.0909  0.1061  0.1136  0.1176  0.1425
holistic        0.0526  0.0526  0.0640  0.0584  0.0670  0.0655

View: readme_and_additional_context
----------------------------------------------------------------------------------------------------
Division        MRR@1   NDCG@1   MRR@5   NDCG@5   MRR@10  NDCG@10
----------------------------------------------------------------------------------------------------
earth           0.0129  0.0129  0.0177  0.0198  0.0192  0.0236
astro           0.2188  0.2188  0.2667  0.2222  0.2667  0.2251
planetary       0.1364  0.1364  0.1606  0.1767  0.1720  0.2054
holistic        0.0574  0.0574  0.0709  0.0673  0.0732  0.0736
====================================================================================================

Legend: MRR = Mean Reciprocal Rank, NDCG = Normalized Discounted Cumulative Gain
```

## Architecture

The benchmark uses **on-the-fly text concatenation**:
- Keeps corpus with all original columns
- Creates text views during benchmarking (not pre-processed)
- More flexible for experimentation
- Easy to add new view combinations

## License

This project benchmarks publicly available NASA datasets. See individual dataset licenses on HuggingFace.

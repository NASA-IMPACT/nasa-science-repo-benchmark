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
  - `full`: All fields combined

- **Multiple Retrieval Methods**:
  - BM25 (sparse retrieval)
  - Embedding-based (dense retrieval with sentence transformers)

- **Comprehensive Metrics**:
  - Recall@K (K=1, 5, 10, 20, 50, 100)
  - NDCG@K (K=1, 5, 10, 20, 50, 100)

## Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
uv sync
```

## Usage

### Run Full Benchmark

Run the complete benchmark with default settings (all views, both retrievers):

```bash
uv run python main.py
```

### Custom Configuration

Run with specific views and retrievers:

```bash
# Test only BM25 with README views
uv run python main.py --retrievers bm25 --views readme readme_cleaned

# Test only embedding retrieval with topic-enhanced views
uv run python main.py --retrievers embedding --views readme_and_topics readme_and_additional_context

# Use a different embedding model
uv run python main.py --embedding-model all-mpnet-base-v2

# Skip visualization plots
uv run python main.py --no-plots

# Save results to custom location
uv run python main.py --output my_results.csv
```

### Available Options

```
--views [VIEW ...]              Text views to benchmark (default: all)
--retrievers [RET ...]          Retrieval methods: bm25, embedding (default: both)
--embedding-model MODEL         Sentence transformer model (default: all-MiniLM-L6-v2)
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
  - 5,397 repositories (corpus)
  - 215 queries
  - 253 relevance judgments

- **Parent Dataset**: [nasa-impact/nasa-science-github-repos](https://huggingface.co/datasets/nasa-impact/nasa-science-github-repos)
  - 5,264 repositories with rich metadata
  - README, cleaned README, topics, additional context, etc.

## Example Output

```
================================================================================
BENCHMARK RESULTS
================================================================================
View                                Retriever     R@10   N@10   R@50   N@50
--------------------------------------------------------------------------------
readme                              bm25          0.456  0.523  0.678  0.612
readme                              embedding     0.489  0.551  0.701  0.634
readme_cleaned                      bm25          0.489  0.551  0.701  0.634
readme_cleaned                      embedding     0.512  0.578  0.723  0.656
readme_and_topics                   bm25          0.512  0.578  0.723  0.656
readme_and_topics                   embedding     0.534  0.601  0.745  0.678
readme_and_additional_context       bm25          0.545  0.612  0.756  0.689
readme_and_additional_context       embedding     0.567  0.634  0.778  0.711
================================================================================

Legend: R@K = Recall@K, N@K = NDCG@K
```

## Architecture

The benchmark uses **on-the-fly text concatenation**:
- Keeps corpus with all original columns
- Creates text views during benchmarking (not pre-processed)
- More flexible for experimentation
- Easy to add new view combinations

## License

This project benchmarks publicly available NASA datasets. See individual dataset licenses on HuggingFace.

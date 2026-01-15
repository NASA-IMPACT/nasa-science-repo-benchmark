---
language:
- en
license: apache-2.0
task_categories:
- text-retrieval
- question-answering
tags:
- information-retrieval
- benchmark
- nasa
- science
- github
- code-search
pretty_name: NASA Science Repos SME Benchmark
size_categories:
- 1K<n<10K
---

# NASA Science Repos SME Benchmark

A benchmark dataset for evaluating retrieval systems on NASA science repository discovery tasks. This dataset contains expert queries, a corpus of NASA science GitHub repositories, and relevance judgments.

## Dataset Description

- **Corpus**: 5,264 NASA science GitHub repositories with complete metadata
- **Queries**: 219 expert-curated questions from subject matter experts
- **Qrels**: 253 relevance judgments across three NASA science divisions
- **Languages**: English
- **License**: Apache 2.0

## Dataset Structure

### Files

```
├── corpus.jsonl          # 5,264 repositories with full metadata
├── queries.jsonl         # 219 expert queries
└── qrels/
    ├── earth.tsv        # Earth Science relevance judgments (162)
    ├── astro.tsv        # Astrophysics relevance judgments (62)
    └── planetary.tsv    # Planetary Science relevance judgments (29)
```

### Data Fields

#### corpus.jsonl

Each line contains a repository with the following fields:

- `_id`: (string) Unique corpus identifier (0-5263)
- `url`: (string) GitHub repository URL
- `name`: (string) Repository name
- `title`: (string) Repository title (alias for name)
- `text`: (string) Repository README content
- `readme`: (string) Raw README content
- `readme_cleaned`: (string) Cleaned/processed README
- `topics`: (string) Repository topics/tags (pipe-separated)
- `division`: (string) NASA Science Mission Directorate division
- `division_reasoning`: (string) Explanation for division classification
- `source`: (string) Discovery source (SDE, ORG, EO-KG, ASCL, KW: Hubble)
- `readme_url`: (string) Direct URL to README file
- `description`: (string) Repository description
- `additional_context`: (string) Enriched contextual information
- `additional_context_reasoning`: (string) Explanation for additional context

#### queries.jsonl

Each line contains a query:

- `_id`: (string) Query identifier (1-219)
- `text`: (string) Query text
- `metadata`: (dict)
  - `division`: (string) NASA division (Earth, Astro, Planetary)
  - `url`: (string) List of relevant repository URLs

#### qrels/*.tsv

Tab-separated files with relevance judgments:

- `query-id`: (int) Query identifier
- `corpus-id`: (int) Corpus document identifier
- `score`: (int) Relevance score (1 = relevant)

## Statistics

### Corpus Distribution

#### By Division
- **Astrophysics Division**: 2,319 repos (44.05%)
- **Earth Science Division**: 2,057 repos (39.08%)
- **Planetary Science Division**: 522 repos (9.92%)
- **Biological and Physical Sciences Division**: 242 repos (4.60%)
- **Heliophysics Division**: 124 repos (2.36%)

#### By Source
- **SDE** (Science Discovery Engine): 2,199 repos (41.77%)
- **ORG** (Curated Organizations): 1,385 repos (26.31%)
- **EO-KG** (Earth Observations Knowledge Graph): 1,339 repos (25.44%)
- **ASCL** (Astrophysics Source Code Library): 320 repos (6.08%)
- **KW: Hubble**: 21 repos (0.40%)

### Query Distribution
- **Earth Science**: 158 queries (72.1%)
- **Astrophysics**: 34 queries (15.5%)
- **Planetary Science**: 27 queries (12.3%)

### Relevance Judgments
- **Total**: 253 query-document pairs
- **Earth Science**: 162 relevance judgments
- **Astrophysics**: 62 relevance judgments
- **Planetary Science**: 29 relevance judgments

## Usage

```python
from datasets import load_dataset

# Load corpus
corpus = load_dataset(
    "nasa-impact/nasa-science-repos-sme-benchmark",
    data_files="corpus.jsonl",
    split="train"
)

# Load queries
queries = load_dataset(
    "nasa-impact/nasa-science-repos-sme-benchmark",
    data_files="queries.jsonl",
    split="train"
)

# Load relevance judgments
qrels_earth = load_dataset(
    "nasa-impact/nasa-science-repos-sme-benchmark",
    data_files="qrels/earth.tsv",
    split="train"
)
```

## Benchmark Tasks

This dataset supports the following tasks:

1. **Ad-hoc Retrieval**: Given a query, retrieve relevant repositories
2. **Code Search**: Find repositories matching natural language descriptions
3. **Question Answering**: Answer questions about NASA science software tools

## Evaluation Metrics

Recommended metrics:
- Recall@K (K=10, 50, 100)
- NDCG@K (K=10, 50, 100)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)

## Changes from Original

This is an updated version of the benchmark with the following improvements:

- **Corpus expanded**: Complete coverage of all 5,264 repositories from the parent dataset
- **Full metadata**: All 15 fields from parent dataset included in corpus
- **Updated queries**: 219 expert queries (from 215)
- **Validated distributions**: All division and source distributions validated

## Source Datasets

- **Parent Dataset**: [nasa-impact/nasa-science-github-repos](https://huggingface.co/datasets/nasa-impact/nasa-science-github-repos)
- **Original Benchmark**: [nasa-impact/nasa-science-repos-sme-benchmark](https://huggingface.co/datasets/nasa-impact/nasa-science-repos-sme-benchmark)

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{nasa_science_repos_sme_benchmark,
  title={NASA Science Repos SME Benchmark},
  author={NASA IMPACT},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/nasa-impact/nasa-science-repos-sme-benchmark}
}
```

## License

Apache License 2.0

## Maintainers

NASA IMPACT Team

## Contact

For questions or issues, please open an issue on the dataset repository.

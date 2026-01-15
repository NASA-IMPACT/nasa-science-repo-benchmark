"""
Convert and update the NASA Science Repos Benchmark dataset.

This script:
1. Creates new corpus.jsonl with ALL 5,264 repos from parent dataset
2. Updates queries.jsonl from local CSV
3. Validates qrels against new corpus
4. Saves locally for validation before pushing to HuggingFace
"""

import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset


def create_corpus_jsonl(output_dir: Path):
    """
    Create corpus.jsonl from parent dataset with all 5,264 repos.
    Includes ALL columns from parent dataset.
    """
    print("Loading parent corpus from HuggingFace...")
    dataset = load_dataset("nasa-impact/nasa-science-github-repos", split="train")
    df = dataset.to_pandas()

    print(f"Loaded {len(df)} repositories")
    print(f"Columns: {df.columns.tolist()}")

    corpus_data = []
    for idx, row in df.iterrows():
        # Create corpus entry with ALL columns
        entry = {"_id": str(idx)}  # Use index as corpus ID (0-5263)

        # Add all columns from parent dataset
        for col in df.columns:
            value = row[col]
            # Convert to JSON-serializable format
            if pd.isna(value):
                entry[col] = None
            elif isinstance(value, list):
                entry[col] = value
            else:
                entry[col] = str(value)

        # Ensure we have 'text' field for compatibility
        if 'text' not in entry:
            entry['text'] = entry.get('readme', '')

        # Ensure we have 'title' field
        if 'title' not in entry:
            entry['title'] = entry.get('name', '')

        corpus_data.append(entry)

    # Save as JSONL
    output_file = output_dir / "corpus.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in corpus_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Created corpus.jsonl with {len(corpus_data)} documents")
    print(f"  Columns per entry: {len(corpus_data[0])}")
    print(f"  Saved to: {output_file}")

    return df, corpus_data


def create_queries_jsonl(output_dir: Path, csv_path: Path):
    """
    Create queries.jsonl from local CSV.

    Format: {"_id": "1", "text": "query text", "metadata": {"division": "..."}}
    """
    print(f"\nLoading queries from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} queries")

    queries_data = []
    for _, row in df.iterrows():
        entry = {
            "_id": str(row['id']),
            "text": str(row['question']),
            "metadata": {
                "division": str(row.get('division', '')),
                "url": str(row.get('url', ''))  # Keep the URL list for reference
            }
        }
        queries_data.append(entry)

    # Save as JSONL
    output_file = output_dir / "queries.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in queries_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Created queries.jsonl with {len(queries_data)} queries")
    print(f"  Saved to: {output_file}")

    return df, queries_data


def validate_qrels(output_dir: Path, corpus_df: pd.DataFrame, queries_df: pd.DataFrame):
    """
    Load existing qrels and validate they match the new corpus/queries.
    """
    print("\nValidating qrels...")

    # Load existing qrels
    qrels_dir = output_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)

    corpus_urls = set(corpus_df['url'].str.strip().str.lower())
    query_ids = set(queries_df['id'].astype(str))

    total_qrels = 0
    missing_corpus = set()
    missing_queries = set()

    for division in ['earth', 'astro', 'planetary']:
        try:
            dataset = load_dataset(
                "nasa-impact/nasa-science-repos-sme-benchmark",
                data_files=f"qrels/{division}.tsv",
                split="train"
            )
            df = dataset.to_pandas()

            # Copy to output
            output_file = qrels_dir / f"{division}.tsv"
            df.to_csv(output_file, sep='\t', index=False)

            total_qrels += len(df)

            # Validate query IDs exist
            for qid in df['query-id'].unique():
                if str(qid) not in query_ids:
                    missing_queries.add(str(qid))

            print(f"  ✓ {division}.tsv: {len(df)} qrels")

        except Exception as e:
            print(f"  ⚠ Could not load qrels/{division}.tsv: {e}")

    print(f"\n✓ Total qrels: {total_qrels}")

    if missing_queries:
        print(f"  ⚠ Warning: {len(missing_queries)} query IDs in qrels but not in queries.jsonl")
        print(f"    Missing query IDs: {sorted(list(missing_queries))[:10]}...")

    return total_qrels


def create_dataset_card(output_dir: Path, stats: dict):
    """Create a README.md with dataset statistics."""
    readme_content = f"""# NASA Science Repos SME Benchmark (Updated)

## Dataset Statistics

- **Corpus**: {stats['corpus_count']} repositories
- **Queries**: {stats['queries_count']} questions
- **Qrels**: {stats['qrels_count']} relevance judgments

## Files

- `corpus.jsonl`: All {stats['corpus_count']} repositories from nasa-science-github-repos
- `queries.jsonl`: {stats['queries_count']} expert queries
- `qrels/earth.tsv`: Earth science relevance judgments
- `qrels/astro.tsv`: Astrophysics relevance judgments
- `qrels/planetary.tsv`: Planetary science relevance judgments

## Changes from Original

- **Corpus expanded**: From 5,397 → {stats['corpus_count']} repositories
- **Queries updated**: From HuggingFace queries.jsonl
- **All repos from parent dataset**: Complete coverage of nasa-science-github-repos

## Source Datasets

- Parent: [nasa-impact/nasa-science-github-repos](https://huggingface.co/datasets/nasa-impact/nasa-science-github-repos)
- Original benchmark: [nasa-impact/nasa-science-repos-sme-benchmark](https://huggingface.co/datasets/nasa-impact/nasa-science-repos-sme-benchmark)
"""

    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"\n✓ Created README.md")


def main():
    print("=" * 80)
    print("NASA Science Repos Benchmark - Dataset Converter")
    print("=" * 80)

    # Setup paths
    output_dir = Path("data/benchmark_updated")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path("data/code_validation_data_v4_normalized_full_dataset.csv")

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    # Create corpus.jsonl
    corpus_df, corpus_data = create_corpus_jsonl(output_dir)

    # Create queries.jsonl
    queries_df, queries_data = create_queries_jsonl(output_dir, csv_path)

    # Validate qrels
    qrels_count = validate_qrels(output_dir, corpus_df, queries_df)

    # Create dataset card
    stats = {
        'corpus_count': len(corpus_data),
        'queries_count': len(queries_data),
        'qrels_count': qrels_count
    }
    create_dataset_card(output_dir, stats)

    print("\n" + "=" * 80)
    print("✓ Dataset conversion complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nFiles created:")
    print(f"  - corpus.jsonl ({len(corpus_data)} repos)")
    print(f"  - queries.jsonl ({len(queries_data)} queries)")
    print(f"  - qrels/*.tsv ({qrels_count} relevance judgments)")
    print(f"  - README.md")
    print("\nNext steps:")
    print("  1. Validate the files in data/benchmark_updated/")
    print("  2. Push to HuggingFace when ready")


if __name__ == "__main__":
    main()

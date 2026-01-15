"""Data loading utilities for NASA Science Repos Benchmark"""

import pandas as pd
from datasets import load_dataset
from typing import Dict, Any
from pathlib import Path


def load_benchmark_corpus() -> pd.DataFrame:
    """
    Load the benchmark corpus from HuggingFace dataset.

    Returns:
        DataFrame with columns: _id, title, text, url
    """
    print("Loading benchmark corpus from HuggingFace...")
    dataset = load_dataset("nasa-impact/nasa-science-repos-sme-benchmark", split="train")

    # Extract corpus.jsonl data
    # The dataset contains corpus data with _id, title, text, url fields
    corpus_data = []

    # Check if the dataset has the corpus data directly or if we need to access it differently
    if hasattr(dataset, 'features') and 'corpus_id' in dataset.features:
        # Data is in the main dataset
        df = dataset.to_pandas()
    else:
        # Load from data_files directly
        dataset = load_dataset(
            "nasa-impact/nasa-science-repos-sme-benchmark",
            data_files="corpus.jsonl",
            split="train"
        )
        df = dataset.to_pandas()

    print(f"Loaded {len(df)} documents from benchmark corpus")
    return df


def load_parent_corpus() -> pd.DataFrame:
    """
    Load the parent corpus with rich metadata from HuggingFace.

    Returns:
        DataFrame with columns: url, readme, readme_cleaned, topics,
        additional_context, description, name, etc.
    """
    print("Loading parent corpus from HuggingFace...")
    dataset = load_dataset("nasa-impact/nasa-science-github-repos", split="train")
    df = dataset.to_pandas()

    print(f"Loaded {len(df)} repositories from parent corpus")
    return df


def enrich_corpus(benchmark_corpus: pd.DataFrame, parent_corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich benchmark corpus with additional fields from parent corpus.

    Args:
        benchmark_corpus: DataFrame from load_benchmark_corpus()
        parent_corpus: DataFrame from load_parent_corpus()

    Returns:
        Enriched DataFrame with all columns from both datasets
    """
    print("Enriching corpus with parent dataset fields...")

    # Ensure URL columns are clean for matching
    benchmark_corpus['url_clean'] = benchmark_corpus['url'].str.strip().str.lower()
    parent_corpus['url_clean'] = parent_corpus['url'].str.strip().str.lower()

    # Left join to keep all benchmark corpus entries
    enriched = benchmark_corpus.merge(
        parent_corpus,
        on='url_clean',
        how='left',
        suffixes=('', '_parent')
    )

    # Clean up duplicate url columns
    if 'url_parent' in enriched.columns:
        enriched['url'] = enriched['url'].fillna(enriched['url_parent'])
        enriched = enriched.drop(columns=['url_parent'])

    enriched = enriched.drop(columns=['url_clean'])

    # Report matching statistics
    matched = enriched['readme'].notna().sum()
    total = len(enriched)
    print(f"Matched {matched}/{total} repos with parent corpus ({matched/total*100:.1f}%)")

    # For unmatched entries, we'll use the 'text' field as fallback for README
    if 'text' in enriched.columns:
        enriched['readme'] = enriched['readme'].fillna(enriched['text'])
        enriched['readme_cleaned'] = enriched['readme_cleaned'].fillna(enriched['text'])

    # Ensure topics is a list
    if 'topics' in enriched.columns:
        enriched['topics'] = enriched['topics'].apply(
            lambda x: x if isinstance(x, list) else []
        )

    print(f"Enriched corpus ready with {len(enriched.columns)} columns")
    return enriched


def load_queries() -> pd.DataFrame:
    """
    Load queries from HuggingFace benchmark dataset with CSV fallback.

    Returns:
        DataFrame with columns: _id, text, metadata
    """
    print("Loading queries...")

    try:
        # Try loading from HuggingFace
        dataset = load_dataset(
            "nasa-impact/nasa-science-repos-sme-benchmark",
            data_files="queries.jsonl",
            split="train"
        )
        df = dataset.to_pandas()
        print(f"Loaded {len(df)} queries from HuggingFace")
        return df
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        print("Trying local CSV fallback...")

        # Fallback to local CSV
        csv_path = Path("data/code_validation_data_v4_normalized_full_dataset.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Rename columns to match expected format
            df = df.rename(columns={'id': '_id', 'question': 'text'})
            # Create metadata from division
            if 'division' in df.columns:
                df['metadata'] = df.apply(lambda row: {'division': row['division']}, axis=1)
            print(f"Loaded {len(df)} queries from local CSV")
            return df
        else:
            raise FileNotFoundError(f"Could not find queries in HuggingFace or local CSV at {csv_path}")


def load_qrels() -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from benchmark dataset.

    Returns:
        Nested dict: {query_id: {corpus_id: score}}
    """
    print("Loading relevance judgments (qrels)...")

    qrels_data = {}

    # Load each qrels file
    for division in ['earth', 'astro', 'planetary']:
        try:
            dataset = load_dataset(
                "nasa-impact/nasa-science-repos-sme-benchmark",
                data_files=f"qrels/{division}.tsv",
                split="train"
            )
            df = dataset.to_pandas()

            # Convert to nested dict format
            for _, row in df.iterrows():
                query_id = str(row['query-id'])
                corpus_id = str(row['corpus-id'])
                score = int(row['score'])

                if query_id not in qrels_data:
                    qrels_data[query_id] = {}
                qrels_data[query_id][corpus_id] = score

            print(f"  Loaded {len(df)} qrels from {division}.tsv")
        except Exception as e:
            print(f"  Warning: Could not load qrels/{division}.tsv: {e}")

    total_qrels = sum(len(v) for v in qrels_data.values())
    print(f"Loaded {len(qrels_data)} queries with {total_qrels} relevance judgments")

    return qrels_data


def get_division_mapping(queries_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create mapping from query_id to division.

    Args:
        queries_df: DataFrame with queries including metadata with division info

    Returns:
        Dict mapping query_id to division name
        e.g., {'1': 'Planetary', '2': 'Earth', ...}
    """
    division_map = {}

    for _, row in queries_df.iterrows():
        query_id = str(row['_id'])
        # Extract division from metadata
        if 'metadata' in row and isinstance(row['metadata'], dict):
            division = row['metadata'].get('division', 'Unknown')
        else:
            division = 'Unknown'

        division_map[query_id] = division

    return division_map


def load_qrels_by_division() -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Load relevance judgments separated by division.

    Returns:
        Dict with division-specific qrels:
        {
            'earth': {query_id: {corpus_id: score}},
            'astro': {query_id: {corpus_id: score}},
            'planetary': {query_id: {corpus_id: score}},
            'holistic': {query_id: {corpus_id: score}}  # all combined
        }
    """
    print("Loading qrels by division...")

    qrels_by_division = {
        'earth': {},
        'astro': {},
        'planetary': {},
        'holistic': {}
    }

    # Load each division's qrels separately
    for division in ['earth', 'astro', 'planetary']:
        try:
            dataset = load_dataset(
                "nasa-impact/nasa-science-repos-sme-benchmark",
                data_files=f"qrels/{division}.tsv",
                split="train"
            )
            df = dataset.to_pandas()

            # Convert to nested dict format
            for _, row in df.iterrows():
                query_id = str(row['query-id'])
                corpus_id = str(row['corpus-id'])
                score = int(row['score'])

                # Add to division-specific qrels
                if query_id not in qrels_by_division[division]:
                    qrels_by_division[division][query_id] = {}
                qrels_by_division[division][query_id][corpus_id] = score

                # Also add to holistic (all divisions combined)
                if query_id not in qrels_by_division['holistic']:
                    qrels_by_division['holistic'][query_id] = {}
                qrels_by_division['holistic'][query_id][corpus_id] = score

            print(f"  Loaded {len(df)} qrels from {division}.tsv ({len(qrels_by_division[division])} queries)")

        except Exception as e:
            print(f"  Warning: Could not load qrels/{division}.tsv: {e}")

    total_queries = len(qrels_by_division['holistic'])
    total_qrels = sum(len(v) for v in qrels_by_division['holistic'].values())
    print(f"Total: {total_queries} queries with {total_qrels} relevance judgments")

    return qrels_by_division

"""Data loading utilities for NASA Science Repos Benchmark"""

import pandas as pd
import json
from datasets import load_dataset
from typing import Dict, Any
from pathlib import Path


# Use local corrected benchmark data (HuggingFace qrels are corrupted)
LOCAL_BENCHMARK_DIR = Path("data/benchmark_updated")


def load_benchmark_corpus() -> pd.DataFrame:
    """
    Load the benchmark corpus.

    Returns:
        DataFrame with columns: _id, title, text, url
    """
    # Use corrected local files
    local_corpus = LOCAL_BENCHMARK_DIR / "corpus.jsonl"
    if local_corpus.exists():
        print(f"Loading benchmark corpus from {local_corpus}...")
        records = []
        with open(local_corpus, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} documents from local corpus")
        return df

    # Fallback to HuggingFace
    print("Loading benchmark corpus from HuggingFace...")
    dataset = load_dataset(
        "nasa-impact/nasa-science-repos-sme-benchmark",
        data_files="corpus.jsonl",
        split="train"
    )
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} documents from HuggingFace corpus")
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
    Load queries from local benchmark or HuggingFace.

    Returns:
        DataFrame with columns: _id, text, metadata
    """
    print("Loading queries...")

    # Use corrected local files
    local_queries = LOCAL_BENCHMARK_DIR / "queries.jsonl"
    if local_queries.exists():
        print(f"Loading queries from {local_queries}...")
        records = []
        with open(local_queries, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} queries from local file")
        return df

    # Fallback to HuggingFace
    try:
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
        raise


def load_qrels() -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from benchmark dataset.

    Returns:
        Nested dict: {query_id: {corpus_id: score}}
    """
    print("Loading relevance judgments (qrels)...")

    qrels_data = {}

    # Use corrected local files
    local_qrels_dir = LOCAL_BENCHMARK_DIR / "qrels"
    if local_qrels_dir.exists():
        for division in ['earth', 'astro', 'planetary']:
            qrels_file = local_qrels_dir / f"{division}.tsv"
            if qrels_file.exists():
                df = pd.read_csv(qrels_file, sep='\t')
                for _, row in df.iterrows():
                    query_id = str(row['query-id'])
                    corpus_id = str(row['corpus-id'])
                    score = int(row['score'])

                    if query_id not in qrels_data:
                        qrels_data[query_id] = {}
                    qrels_data[query_id][corpus_id] = score

                print(f"  Loaded {len(df)} qrels from local {division}.tsv")

        if qrels_data:
            total_qrels = sum(len(v) for v in qrels_data.values())
            print(f"Loaded {len(qrels_data)} queries with {total_qrels} relevance judgments")
            return qrels_data

    # Fallback to HuggingFace
    for division in ['earth', 'astro', 'planetary']:
        try:
            dataset = load_dataset(
                "nasa-impact/nasa-science-repos-sme-benchmark",
                data_files=f"qrels/{division}.tsv",
                split="train"
            )
            df = dataset.to_pandas()

            for _, row in df.iterrows():
                query_id = str(row['query-id'])
                corpus_id = str(row['corpus-id'])
                score = int(row['score'])

                if query_id not in qrels_data:
                    qrels_data[query_id] = {}
                qrels_data[query_id][corpus_id] = score

            print(f"  Loaded {len(df)} qrels from HuggingFace {division}.tsv")
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

    # Use corrected local files
    local_qrels_dir = LOCAL_BENCHMARK_DIR / "qrels"
    if local_qrels_dir.exists():
        for division in ['earth', 'astro', 'planetary']:
            qrels_file = local_qrels_dir / f"{division}.tsv"
            if qrels_file.exists():
                df = pd.read_csv(qrels_file, sep='\t')
                for _, row in df.iterrows():
                    query_id = str(row['query-id'])
                    corpus_id = str(row['corpus-id'])
                    score = int(row['score'])

                    if query_id not in qrels_by_division[division]:
                        qrels_by_division[division][query_id] = {}
                    qrels_by_division[division][query_id][corpus_id] = score

                    if query_id not in qrels_by_division['holistic']:
                        qrels_by_division['holistic'][query_id] = {}
                    qrels_by_division['holistic'][query_id][corpus_id] = score

                print(f"  Loaded {len(df)} qrels from local {division}.tsv ({len(qrels_by_division[division])} queries)")

        if qrels_by_division['holistic']:
            total_queries = len(qrels_by_division['holistic'])
            total_qrels = sum(len(v) for v in qrels_by_division['holistic'].values())
            print(f"Total: {total_queries} queries with {total_qrels} relevance judgments")
            return qrels_by_division

    # Fallback to HuggingFace
    for division in ['earth', 'astro', 'planetary']:
        try:
            dataset = load_dataset(
                "nasa-impact/nasa-science-repos-sme-benchmark",
                data_files=f"qrels/{division}.tsv",
                split="train"
            )
            df = dataset.to_pandas()

            for _, row in df.iterrows():
                query_id = str(row['query-id'])
                corpus_id = str(row['corpus-id'])
                score = int(row['score'])

                if query_id not in qrels_by_division[division]:
                    qrels_by_division[division][query_id] = {}
                qrels_by_division[division][query_id][corpus_id] = score

                if query_id not in qrels_by_division['holistic']:
                    qrels_by_division['holistic'][query_id] = {}
                qrels_by_division['holistic'][query_id][corpus_id] = score

            print(f"  Loaded {len(df)} qrels from HuggingFace {division}.tsv ({len(qrels_by_division[division])} queries)")

        except Exception as e:
            print(f"  Warning: Could not load qrels/{division}.tsv: {e}")

    total_queries = len(qrels_by_division['holistic'])
    total_qrels = sum(len(v) for v in qrels_by_division['holistic'].values())
    print(f"Total: {total_queries} queries with {total_qrels} relevance judgments")

    return qrels_by_division

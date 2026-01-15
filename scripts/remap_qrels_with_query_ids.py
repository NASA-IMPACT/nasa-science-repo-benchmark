"""
Remap BOTH corpus IDs and query IDs in qrels to match the new dataset.

The issue:
- Old qrels use 0-indexed query IDs (0-214)
- New queries use 1-indexed query IDs (1-219) from CSV
- Old qrels use old corpus IDs
- New corpus uses IDs 0-5263

This script remaps both.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path


def main():
    print("=" * 80)
    print("Remap Qrels: Both Query IDs and Corpus IDs")
    print("=" * 80)

    # Load new corpus for URL mapping
    print("\n[1/3] Loading new corpus...")
    new_corpus = load_dataset(
        "nasa-impact/nasa-science-repos-sme-benchmark",
        data_files="corpus.jsonl",
        split="train"
    ).to_pandas()

    url_to_new_corpus_id = {}
    for _, row in new_corpus.iterrows():
        url = str(row['url']).strip().lower()
        url_to_new_corpus_id[url] = str(row['_id'])

    print(f"  New corpus: {len(new_corpus)} repos")

    # Load new queries
    print("\n[2/3] Loading new queries...")
    new_queries = load_dataset(
        "nasa-impact/nasa-science-repos-sme-benchmark",
        data_files="queries.jsonl",
        split="train"
    ).to_pandas()

    # The queries have _id field (1-219) and we need to create mapping from old query index to new query ID
    # Problem: we need to match old queries to new queries
    # The CSV has 219 queries with IDs 1-219
    # The old benchmark had 215 queries with IDs 0-214
    # We can match by question text

    print(f"  New queries: {len(new_queries)}")

    # Create mapping: old_query_id -> new_query_id based on text matching
    # First, load the local CSV to understand the structure
    csv_path = Path("data/code_validation_data_v4_normalized_full_dataset.csv")
    csv_queries = pd.read_csv(csv_path)

    # The CSV has id=1-219, questions
    # The qrels reference the OLD query IDs (0-214 presumably)
    # We need to match qrels query-id to CSV id

    # Actually, the qrels query IDs might already map to CSV IDs if they were generated from the CSV
    # Let me just check: qrels have query-id 0-35 for planetary
    # But CSV has id 1-219
    # So qrels query-id 0 might actually mean CSV id 1 (just off by 1)

    # Let me try: old_query_id + 1 = new_query_id
    # This is a simple offset correction

    print("\n[3/3] Remapping qrels...")

    output_dir = Path("data/benchmark_updated/qrels")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_remapped = 0
    total_unmatched_corpus = 0
    total_unmatched_query = 0

    for division in ['earth', 'astro', 'planetary']:
        print(f"\nProcessing {division}.tsv...")

        # Load qrels
        qrels = load_dataset(
            "nasa-impact/nasa-science-repos-sme-benchmark",
            data_files=f"qrels/{division}.tsv",
            split="train"
        ).to_pandas()

        print(f"  Original: {len(qrels)} qrels")

        # Load old corpus for URL lookup
        old_corpus = load_dataset(
            "nasa-impact/nasa-science-repos-sme-benchmark",
            data_files="corpus.jsonl",
            split="train"
        ).to_pandas()

        old_corpus_id_to_url = {}
        for _, row in old_corpus.iterrows():
            old_corpus_id_to_url[str(row['_id'])] = str(row['url']).strip().lower()

        # Remap
        remapped = []
        unmatched_corpus = 0
        unmatched_query = 0

        for _, row in qrels.iterrows():
            old_query_id = int(row['query-id'])
            old_corpus_id = str(row['corpus-id'])
            score = int(row['score'])

            # Remap query ID: old_query_id + 1 = new_query_id (0->1, 1->2, etc.)
            new_query_id = old_query_id + 1

            # Check if new query ID exists (convert to string for comparison)
            if str(new_query_id) not in new_queries['_id'].values:
                unmatched_query += 1
                continue

            # Remap corpus ID via URL
            if old_corpus_id in old_corpus_id_to_url:
                url = old_corpus_id_to_url[old_corpus_id]
                if url in url_to_new_corpus_id:
                    new_corpus_id = int(url_to_new_corpus_id[url])
                    remapped.append({
                        'query-id': new_query_id,
                        'corpus-id': new_corpus_id,
                        'score': score
                    })
                else:
                    unmatched_corpus += 1
            else:
                unmatched_corpus += 1

        remapped_df = pd.DataFrame(remapped)
        output_file = output_dir / f"{division}.tsv"
        remapped_df.to_csv(output_file, sep='\t', index=False)

        print(f"  Remapped: {len(remapped_df)} qrels")
        print(f"  Unmatched corpus: {unmatched_corpus}")
        print(f"  Unmatched query: {unmatched_query}")
        print(f"  ✓ Saved to: {output_file}")

        total_remapped += len(remapped_df)
        total_unmatched_corpus += unmatched_corpus
        total_unmatched_query += unmatched_query

    print("\n" + "=" * 80)
    print("✓ Remapping Complete!")
    print("=" * 80)
    print(f"Total remapped: {total_remapped}")
    print(f"Total unmatched corpus: {total_unmatched_corpus}")
    print(f"Total unmatched query: {total_unmatched_query}")
    print(f"\nRemapped files: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

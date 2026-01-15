"""
Remap qrels corpus IDs to match the new corpus.

The new corpus has IDs 0-5263 based on the full parent dataset.
The old qrels reference corpus IDs from the previous benchmark corpus.
This script remaps the qrels using URL matching.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path


def create_corpus_id_mapping():
    """
    Create mapping from old corpus IDs to new corpus IDs based on URL matching.

    Returns:
        Dict mapping old_corpus_id -> new_corpus_id
    """
    print("Creating corpus ID mapping...")

    # Load NEW corpus (5,264 repos with IDs 0-5263)
    print("  Loading new corpus...")
    new_corpus_dataset = load_dataset(
        "nasa-impact/nasa-science-repos-sme-benchmark",
        data_files="corpus.jsonl",
        split="train"
    )
    new_corpus = new_corpus_dataset.to_pandas()

    # Create URL -> new_id mapping
    url_to_new_id = {}
    for _, row in new_corpus.iterrows():
        url = str(row['url']).strip().lower()
        new_id = str(row['_id'])
        url_to_new_id[url] = new_id

    print(f"  New corpus: {len(new_corpus)} repos")
    print(f"  URL mapping created: {len(url_to_new_id)} URLs")

    return url_to_new_id


def remap_qrels_file(division: str, url_to_new_id: dict, output_dir: Path):
    """
    Remap qrels for a specific division.

    Args:
        division: Division name (earth, astro, planetary)
        url_to_new_id: Mapping from URL to new corpus ID
        output_dir: Directory to save updated qrels
    """
    print(f"\nProcessing {division}.tsv...")

    # Load qrels for this division
    qrels_dataset = load_dataset(
        "nasa-impact/nasa-science-repos-sme-benchmark",
        data_files=f"qrels/{division}.tsv",
        split="train"
    )
    qrels_df = qrels_dataset.to_pandas()

    print(f"  Original qrels: {len(qrels_df)} entries")

    # Load old corpus to get URL for each old corpus-id
    print("  Loading old benchmark corpus for URL lookup...")
    # The qrels reference the original corpus.jsonl structure
    old_corpus_dataset = load_dataset(
        "nasa-impact/nasa-science-repos-sme-benchmark",
        data_files="corpus.jsonl",
        split="train"
    )
    old_corpus = old_corpus_dataset.to_pandas()

    # Create old_id -> URL mapping
    old_id_to_url = {}
    for _, row in old_corpus.iterrows():
        old_id = str(row['_id'])
        url = str(row['url']).strip().lower()
        old_id_to_url[old_id] = url

    # Remap corpus IDs
    remapped_qrels = []
    unmatched = []

    for _, row in qrels_df.iterrows():
        query_id = row['query-id']
        old_corpus_id = str(row['corpus-id'])
        score = row['score']

        # Get URL from old corpus ID
        if old_corpus_id in old_id_to_url:
            url = old_id_to_url[old_corpus_id]

            # Get new corpus ID from URL
            if url in url_to_new_id:
                new_corpus_id = url_to_new_id[url]
                remapped_qrels.append({
                    'query-id': query_id,
                    'corpus-id': int(new_corpus_id),
                    'score': score
                })
            else:
                unmatched.append({
                    'old_corpus_id': old_corpus_id,
                    'url': url,
                    'reason': 'URL not found in new corpus'
                })
        else:
            unmatched.append({
                'old_corpus_id': old_corpus_id,
                'url': 'N/A',
                'reason': 'Old corpus ID not found'
            })

    # Create remapped DataFrame
    remapped_df = pd.DataFrame(remapped_qrels)

    print(f"  Remapped: {len(remapped_df)} entries")
    print(f"  Unmatched: {len(unmatched)} entries")

    if unmatched:
        print(f"  Warning: {len(unmatched)} corpus IDs could not be remapped")
        print(f"  First few unmatched:")
        for item in unmatched[:5]:
            print(f"    - Old ID: {item['old_corpus_id']}, URL: {item['url']}, Reason: {item['reason']}")

    # Save remapped qrels
    output_file = output_dir / f"{division}.tsv"
    remapped_df.to_csv(output_file, sep='\t', index=False)
    print(f"  ✓ Saved to: {output_file}")

    return len(remapped_df), len(unmatched)


def main():
    print("=" * 80)
    print("Remap Qrels to New Corpus IDs")
    print("=" * 80)

    # Create output directory
    output_dir = Path("data/benchmark_updated/qrels")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create corpus ID mapping
    url_to_new_id = create_corpus_id_mapping()

    # Remap each division
    total_remapped = 0
    total_unmatched = 0

    for division in ['earth', 'astro', 'planetary']:
        remapped, unmatched = remap_qrels_file(division, url_to_new_id, output_dir)
        total_remapped += remapped
        total_unmatched += unmatched

    print("\n" + "=" * 80)
    print("Remapping Complete!")
    print("=" * 80)
    print(f"Total remapped: {total_remapped} qrels")
    print(f"Total unmatched: {total_unmatched} qrels")
    print(f"\nRemapped files saved to: {output_dir.absolute()}")

    if total_unmatched > 0:
        print(f"\n⚠ Warning: {total_unmatched} qrels could not be remapped")
        print("This may be because some repos in the old corpus are not in the new corpus")
    else:
        print("\n✓ All qrels successfully remapped!")


if __name__ == "__main__":
    main()

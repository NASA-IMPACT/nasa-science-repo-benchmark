"""
Push updated benchmark dataset to HuggingFace.

Uploads the updated dataset files to:
https://huggingface.co/datasets/nasa-impact/nasa-science-repos-sme-benchmark
"""

from pathlib import Path
from huggingface_hub import HfApi, login
import os


def push_dataset(dataset_dir: Path, repo_id: str = "nasa-impact/nasa-science-repos-sme-benchmark"):
    """
    Push dataset files to HuggingFace.

    Args:
        dataset_dir: Local directory with dataset files
        repo_id: HuggingFace dataset repository ID
    """
    print("=" * 80)
    print("Push Dataset to HuggingFace")
    print("=" * 80)
    print(f"\nRepository: {repo_id}")
    print(f"Source directory: {dataset_dir.absolute()}\n")

    # Check if files exist
    required_files = [
        "corpus.jsonl",
        "queries.jsonl",
        "README.md"
    ]

    qrels_files = list((dataset_dir / "qrels").glob("*.tsv"))

    print("Checking files...")
    for file in required_files:
        file_path = dataset_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {file} NOT FOUND")
            return

    print(f"  ✓ qrels/ ({len(qrels_files)} files)")
    for qrel_file in qrels_files:
        print(f"    - {qrel_file.name}")

    # Login to HuggingFace
    print("\n" + "-" * 80)
    print("Authentication")
    print("-" * 80)

    # Check if already logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Already logged in as: {user['name']}")
    except Exception:
        print("Not logged in. Please authenticate...")
        print("\nYou can either:")
        print("  1. Set HF_TOKEN environment variable")
        print("  2. Run: huggingface-cli login")
        print("  3. Use token parameter in this script")

        # Try to login
        token = os.environ.get("HF_TOKEN")
        if token:
            print("Found HF_TOKEN in environment, logging in...")
            login(token=token)
            print("✓ Logged in successfully")
        else:
            print("\n⚠ No HF_TOKEN found. Please login first:")
            print("  export HF_TOKEN='your_token_here'")
            print("  OR")
            print("  huggingface-cli login")
            return

    # Confirm before pushing
    print("\n" + "-" * 80)
    print("Ready to push")
    print("-" * 80)
    print(f"\nThis will upload to: https://huggingface.co/datasets/{repo_id}")
    print("\nFiles to upload:")
    print("  - corpus.jsonl (5,264 repos, ~79MB)")
    print("  - queries.jsonl (219 queries)")
    print("  - qrels/*.tsv (253 judgments)")
    print("  - README.md")

    response = input("\nProceed with upload? (yes/no): ").strip().lower()
    if response != "yes":
        print("Upload cancelled.")
        return

    # Upload files
    print("\n" + "-" * 80)
    print("Uploading files...")
    print("-" * 80)

    api = HfApi()

    try:
        # Upload corpus.jsonl
        print("\n[1/4] Uploading corpus.jsonl...")
        api.upload_file(
            path_or_fileobj=str(dataset_dir / "corpus.jsonl"),
            path_in_repo="corpus.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update corpus with all 5,264 repos and full metadata"
        )
        print("  ✓ corpus.jsonl uploaded")

        # Upload queries.jsonl
        print("\n[2/4] Uploading queries.jsonl...")
        api.upload_file(
            path_or_fileobj=str(dataset_dir / "queries.jsonl"),
            path_in_repo="queries.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update queries with 219 questions"
        )
        print("  ✓ queries.jsonl uploaded")

        # Upload qrels files
        print("\n[3/4] Uploading qrels/*.tsv...")
        for qrel_file in qrels_files:
            api.upload_file(
                path_or_fileobj=str(qrel_file),
                path_in_repo=f"qrels/{qrel_file.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update {qrel_file.name}"
            )
            print(f"  ✓ qrels/{qrel_file.name} uploaded")

        # Upload README
        print("\n[4/4] Uploading README.md...")
        api.upload_file(
            path_or_fileobj=str(dataset_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update README with new dataset statistics"
        )
        print("  ✓ README.md uploaded")

        print("\n" + "=" * 80)
        print("✓ Upload complete!")
        print("=" * 80)
        print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
        print("\nThe dataset is now updated with:")
        print("  - 5,264 repositories (full parent dataset)")
        print("  - 219 queries")
        print("  - 253 relevance judgments")
        print("  - Complete metadata (15 columns per repo)")

    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        print("\nPlease check:")
        print("  1. You have write access to the repository")
        print("  2. Your authentication token is valid")
        print("  3. Network connection is stable")


def main():
    dataset_dir = Path("data/benchmark_updated")

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please run convert_benchmark_dataset.py first")
        return

    push_dataset(dataset_dir)


if __name__ == "__main__":
    main()

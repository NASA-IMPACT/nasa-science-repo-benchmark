"""Main benchmark runner"""

import pandas as pd
from typing import List, Dict
from .data_loader import load_benchmark_corpus, load_parent_corpus, enrich_corpus, load_queries, load_qrels
from .view_generator import ViewGenerator
from .retrievers import BM25Retriever, EmbeddingRetriever
from .metrics import evaluate_retrieval


def run_benchmark(
    view_names: List[str] = None,
    retriever_names: List[str] = None,
    k_values: List[int] = None,
    embedding_model: str = 'all-MiniLM-L6-v2'
) -> pd.DataFrame:
    """
    Run complete benchmark across views and retrievers.

    Args:
        view_names: List of view names to test
        retriever_names: List of retriever names ('bm25', 'embedding')
        k_values: K values for metrics evaluation
        embedding_model: Model name for embedding retriever

    Returns:
        DataFrame with benchmark results
    """
    # Defaults
    if view_names is None:
        view_names = [
            'readme',
            'readme_cleaned',
            'readme_and_topics',
            'readme_and_additional_context'
        ]

    if retriever_names is None:
        retriever_names = ['bm25', 'embedding']

    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]

    print("=" * 80)
    print("NASA Science Repos Benchmark")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    benchmark_corpus = load_benchmark_corpus()
    parent_corpus = load_parent_corpus()
    corpus = enrich_corpus(benchmark_corpus, parent_corpus)
    queries = load_queries()
    qrels = load_qrels()

    print(f"\nData loaded:")
    print(f"  - Corpus: {len(corpus)} documents")
    print(f"  - Queries: {len(queries)} queries")
    print(f"  - Qrels: {len(qrels)} query-document pairs")

    # Initialize view generator
    view_gen = ViewGenerator()

    # Store all results
    all_results = []

    # Run benchmark for each combination
    total_combinations = len(view_names) * len(retriever_names)
    current = 0

    for view_name in view_names:
        print(f"\n[2/5] Creating view: {view_name}")
        corpus_view = view_gen.create_text_view(corpus, view_name)

        for retriever_name in retriever_names:
            current += 1
            print(f"\n[3/5] Running retriever [{current}/{total_combinations}]: {retriever_name} on {view_name}")

            # Initialize retriever
            if retriever_name == 'bm25':
                retriever = BM25Retriever()
            elif retriever_name == 'embedding':
                retriever = EmbeddingRetriever(model_name=embedding_model)
            else:
                print(f"Unknown retriever: {retriever_name}, skipping...")
                continue

            # Index corpus
            print("[4/5] Indexing...")
            retriever.index(corpus_view)

            # Search
            print("[5/5] Searching...")
            search_results = retriever.search(queries, top_k=max(k_values))

            # Evaluate
            print("Evaluating...")
            metrics = evaluate_retrieval(qrels, search_results, k_values=k_values)

            # Store results
            result_row = {
                'view': view_name,
                'retriever': retriever_name,
                **metrics
            }
            all_results.append(result_row)

            # Print summary
            print(f"\nResults for {view_name} + {retriever_name}:")
            for k in [10, 50]:
                if f'recall@{k}' in metrics:
                    print(f"  Recall@{k:3d}: {metrics[f'recall@{k}']:.4f}")
                if f'ndcg@{k}' in metrics:
                    print(f"  NDCG@{k:3d}:   {metrics[f'ndcg@{k}']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

    return results_df


def format_results_table(results_df: pd.DataFrame, k_values: List[int] = None) -> str:
    """
    Format results as a nice table for display.

    Args:
        results_df: Results DataFrame from run_benchmark()
        k_values: K values to include in table

    Returns:
        Formatted table string
    """
    if k_values is None:
        k_values = [10, 50]

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 100)

    # Header
    header = f"{'View':<35} {'Retriever':<12}"
    for k in k_values:
        header += f" R@{k:<3} "
        header += f" N@{k:<3} "
    lines.append(header)
    lines.append("-" * 100)

    # Rows
    for _, row in results_df.iterrows():
        line = f"{row['view']:<35} {row['retriever']:<12}"
        for k in k_values:
            recall_key = f'recall@{k}'
            ndcg_key = f'ndcg@{k}'
            if recall_key in row:
                line += f" {row[recall_key]:.3f} "
            else:
                line += " ----- "
            if ndcg_key in row:
                line += f" {row[ndcg_key]:.3f} "
            else:
                line += " ----- "
        lines.append(line)

    lines.append("=" * 100)
    lines.append("\nLegend: R@K = Recall@K, N@K = NDCG@K")

    return "\n".join(lines)


def run_bm25_benchmark_by_division(
    view_names: List[str] = None,
    k_values: List[int] = None
) -> pd.DataFrame:
    """
    Run BM25 benchmark with division-specific evaluation.

    Args:
        view_names: List of view names to test
        k_values: K values for metrics evaluation (default: [1, 5, 10])

    Returns:
        DataFrame with columns: view, division, mrr@1, mrr@5, mrr@10,
                                ndcg@1, ndcg@5, ndcg@10
    """
    # Defaults
    if view_names is None:
        view_names = [
            'readme',
            'readme_cleaned',
            'readme_and_topics',
            'readme_and_additional_context'
        ]

    if k_values is None:
        k_values = [1, 5, 10]

    print("=" * 80)
    print("BM25 Benchmark by Division")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    from .data_loader import load_benchmark_corpus, load_parent_corpus, enrich_corpus, load_queries, load_qrels_by_division

    benchmark_corpus = load_benchmark_corpus()
    parent_corpus = load_parent_corpus()
    corpus = enrich_corpus(benchmark_corpus, parent_corpus)
    queries = load_queries()
    qrels_by_division = load_qrels_by_division()

    print(f"\nData loaded:")
    print(f"  - Corpus: {len(corpus)} documents")
    print(f"  - Queries: {len(queries)} queries")
    print(f"  - Divisions: {list(qrels_by_division.keys())}")

    # Initialize view generator
    view_gen = ViewGenerator()

    # Store all results
    all_results = []

    # Run benchmark for each view
    total_views = len(view_names)
    for idx, view_name in enumerate(view_names, 1):
        print(f"\n[2/4] Creating view [{idx}/{total_views}]: {view_name}")
        corpus_view = view_gen.create_text_view(corpus, view_name)

        # Initialize BM25 retriever
        print("[3/4] Indexing with BM25...")
        retriever = BM25Retriever()
        retriever.index(corpus_view)

        # Search all queries
        print("[4/4] Searching...")
        search_results = retriever.search(queries, top_k=max(k_values))

        # Evaluate for each division
        print("Evaluating by division...")
        for division in ['earth', 'astro', 'planetary', 'holistic']:
            division_qrels = qrels_by_division[division]

            # Filter search results to only include queries in this division
            division_results = {qid: res for qid, res in search_results.items()
                               if qid in division_qrels}

            # Calculate metrics
            metrics = evaluate_retrieval(division_qrels, division_results, k_values=k_values)

            # Store result
            result_row = {
                'view': view_name,
                'division': division,
                **metrics
            }
            all_results.append(result_row)

            # Print summary
            print(f"  {division:12s}: MRR@10={metrics.get('mrr@10', 0):.4f}, NDCG@10={metrics.get('ndcg@10', 0):.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

    return results_df


def format_results_table_by_division(results_df: pd.DataFrame, k_values: List[int] = None) -> str:
    """
    Format division-specific results as a nice table for display.

    Args:
        results_df: Results DataFrame from run_bm25_benchmark_by_division()
        k_values: K values to include in table

    Returns:
        Formatted table string
    """
    if k_values is None:
        k_values = [1, 5, 10]

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BM25 BENCHMARK RESULTS BY DIVISION")
    lines.append("=" * 100)

    # Group by view
    for view_name in results_df['view'].unique():
        view_df = results_df[results_df['view'] == view_name]

        lines.append(f"\nView: {view_name}")
        lines.append("-" * 100)

        # Header
        header = f"{'Division':<15}"
        for k in k_values:
            header += f" MRR@{k:<2} "
            header += f" NDCG@{k:<2} "
        lines.append(header)
        lines.append("-" * 100)

        # Rows for each division
        for _, row in view_df.iterrows():
            line = f"{row['division']:<15}"
            for k in k_values:
                mrr_key = f'mrr@{k}'
                ndcg_key = f'ndcg@{k}'
                if mrr_key in row:
                    line += f" {row[mrr_key]:.4f} "
                else:
                    line += " ------ "
                if ndcg_key in row:
                    line += f" {row[ndcg_key]:.4f} "
                else:
                    line += " ------ "
            lines.append(line)

    lines.append("\n" + "=" * 100)
    lines.append("Legend: MRR = Mean Reciprocal Rank, NDCG = Normalized Discounted Cumulative Gain")

    return "\n".join(lines)

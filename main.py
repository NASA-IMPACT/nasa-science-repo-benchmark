"""
NASA Science Repos Benchmark

Run benchmarks to compare different text views (readme, readme_cleaned, readme+topics, etc.)
with different retrieval methods (BM25, embedding-based).
"""

import argparse
from pathlib import Path
from src.benchmark import run_benchmark, format_results_table, run_bm25_benchmark_by_division, format_results_table_by_division
from src.visualize import create_comparison_plots


def main():
    parser = argparse.ArgumentParser(
        description="NASA Science Repos Benchmark - Compare retrieval methods across text views"
    )
    parser.add_argument(
        '--views',
        nargs='+',
        default=['readme', 'readme_cleaned', 'readme_and_topics', 'readme_and_additional_context'],
        help='Text views to benchmark'
    )
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=['bm25', 'embedding'],
        help='Retrieval methods to test (bm25, embedding)'
    )
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embedding retriever'
    )
    parser.add_argument(
        '--output',
        default='results/benchmark_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    parser.add_argument(
        '--by-division',
        action='store_true',
        help='Run BM25 benchmark with division-specific evaluation (Earth, Astro, Planetary, Holistic)'
    )

    args = parser.parse_args()

    # Check if running division-specific benchmark
    if args.by_division:
        # Run BM25 benchmark by division
        results_df = run_bm25_benchmark_by_division(
            view_names=args.views,
            k_values=[1, 5, 10]
        )

        # Display formatted results
        print(format_results_table_by_division(results_df))

        # Save results
        output_path = Path('results/benchmark_results_by_division.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path.absolute()}")

    else:
        # Run standard benchmark
        results_df = run_benchmark(
            view_names=args.views,
            retriever_names=args.retrievers,
            embedding_model=args.embedding_model
        )

        # Display formatted results
        print(format_results_table(results_df))

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path.absolute()}")

        # Create visualizations
        if not args.no_plots:
            create_comparison_plots(results_df, output_dir='results/plots')

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()

"""
NASA Science Repos Benchmark

Run benchmarks to compare different text views (readme, readme_cleaned, readme+topics, etc.)
with different retrieval methods (BM25, embedding-based).
"""

import argparse
from pathlib import Path
from src.benchmark import run_benchmark, format_results_table, run_benchmark_by_division, format_results_table_by_division
from src.visualize import create_comparison_plots


def main():
    parser = argparse.ArgumentParser(
        description="NASA Science Repos Benchmark - Compare retrieval methods across text views"
    )
    parser.add_argument(
        '--views',
        nargs='+',
        default=['readme'],
        help='Text views to benchmark (readme, readme_cleaned, readme_and_topics, readme_and_additional_context). Default: readme only.'
    )
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=['bm25'],
        help='Retrieval methods to test (bm25, embedding, hybrid-rrf, hybrid-rerank). Default: bm25 only.'
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
        # Run benchmark by division for each retriever
        for retriever_name in args.retrievers:
            results_df = run_benchmark_by_division(
                retriever_name=retriever_name,
                embedding_model=args.embedding_model,
                view_names=args.views,
                k_values=[1, 5, 10]
            )

            # Display formatted results
            print(format_results_table_by_division(
                results_df,
                retriever_name=retriever_name,
                embedding_model=args.embedding_model
            ))

            # Save results with dynamic filename
            if retriever_name == 'bm25':
                output_filename = 'bm25_benchmark_results_by_division.csv'
            elif retriever_name == 'embedding':
                model_suffix = args.embedding_model.replace('/', '-').replace('\\', '-')
                output_filename = f'embedding_{model_suffix}_benchmark_results_by_division.csv'
            elif retriever_name == 'hybrid-rrf':
                model_suffix = args.embedding_model.replace('/', '-').replace('\\', '-')
                output_filename = f'hybrid-rrf_{model_suffix}_benchmark_results_by_division.csv'
            elif retriever_name == 'hybrid-rerank':
                model_suffix = args.embedding_model.replace('/', '-').replace('\\', '-')
                output_filename = f'hybrid-rerank_{model_suffix}_benchmark_results_by_division.csv'
            else:
                output_filename = f'{retriever_name}_benchmark_results_by_division.csv'

            output_path = Path('results') / output_filename
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

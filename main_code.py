import os
from typing import List, Dict, Any
import pandas as pd
import argparse
from pathlib import Path
from datasets import load_dataset, load_dataset_builder, get_dataset_config_names
from pathlib import PurePosixPath
from dotenv import load_dotenv
from src.benchmark import format_results_table_by_category
from src.metrics_v2 import evaluate_retrieval
from src.visualize import create_comparison_plots
from src.retrievers import BM25Retriever_V2 as BM25Retriever, EmbeddingRetriever


load_dotenv()

def get_command_arguments():
    parser = argparse.ArgumentParser(
        description="NASA Science Code Benchmark - Compare retrieval methods across text views"
    )
    parser.add_argument(
        '--benchamark-path',
        type=str,
        default="nasa-impact/nasa-science-code-benchmark-v0.1.1",
        required=False,
        help='Path to the benchmark directory containing code repositories'
    )
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=['bm25'],
        choices=['bm25', 'embedding'],
        help='Retrieval methods to test (bm25, embedding). Default: bm25 only.'
    )
    parser.add_argument(
        '--embedding-model',
        default='nasa-impact/indus-sde-st-v0.2',
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
        "--ks", 
        nargs="*", 
        default=[1, 5, 10]
        )
    parser.add_argument(
        '--by-division',
        action='store_true',
        help='Run division-specific evaluation (Earth, Astro, Planetary, Holistic)'
    )
    parser.add_argument(
        '--by-language',
        action='store_true',
        help='Run language-specific evaluation'
    )
    parser.add_argument(
        '--by-query-type',
        action='store_true',
        help='Run query-type-specific evaluation'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for embedding encoding'
    )

    return parser.parse_args()


def get_benchmark_dataset(benchmark_path: str):

    corpus = load_dataset(benchmark_path, data_files="corpus.jsonl", split="train")
    queries = load_dataset(benchmark_path, data_files="queries.jsonl", split="train")

    return corpus, queries

def _dataset_to_qrels_dict(qrels_ds):
    """Convert HF Dataset to {query_id: {corpus_id: relevance_score}}"""
    qrels = {}
    for row in qrels_ds:
        qid = str(row['query-id'])
        cid = str(row['corpus-id'])
        score = int(row['score'])
        
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = score
    
    return qrels

def load_qrels_by_category(benchmark_path, config_name):
    qrels_dict = {}
    # get the categories in the config name
    builder = load_dataset_builder(benchmark_path, name=config_name)
    data_files = builder.config.data_files or {}
    file_list = data_files.get("test") or []

    if not file_list:
        raise ValueError("No 'test' qrels files found for the 'division' config.")
    categories = [PurePosixPath(p).stem for p in file_list]

    for category in categories:
        file_name = f"qrels/{config_name}/{category}.tsv"
        category_qrels_ds = load_dataset(benchmark_path, data_files=file_name, delimiter="\t", split="train")
        # Convert to dict format expected by evaluate_retrieval
        qrels_dict[category] = _dataset_to_qrels_dict(category_qrels_ds)
    
    return qrels_dict


def load_qrels_overall(benchmark_path):
    # need to select a random config name and use all the files in them
    config_names = get_dataset_config_names(benchmark_path)
    if len(config_names) == 0:
        raise ValueError("No config names found in the dataset.")
    config_name = config_names[0]

    # get the categories in the config name
    builder = load_dataset_builder(benchmark_path, name=config_name)
    data_files = builder.config.data_files or {}
    file_list = data_files.get("test") or []

    if not file_list:
        raise ValueError("No 'test' qrels files found for the 'division' config.")
    categories = [PurePosixPath(p).stem for p in file_list]

    # Combine all qrels into one dict
    combined_qrels = {}

    for category in categories:
        file_name = f"qrels/{config_name}/{category}.tsv"
        category_qrels_ds = load_dataset(benchmark_path, data_files=file_name, delimiter="\t", split="train")
        
        # Merge into combined_qrels
        for row in category_qrels_ds:
            qid = str(row['query-id'])
            cid = str(row['corpus-id'])
            score = int(row['score'])
            
            if qid not in combined_qrels:
                combined_qrels[qid] = {}
            combined_qrels[qid][cid] = score
    
    return combined_qrels


def run_benchmark_by_config(
    config_name: str,
    benchmark_path: str,
    retrievers: Dict[str, Any],
    queries_df: pd.DataFrame,
    retriever_names: List[str],
    embedding_model: str,
    k_values: List[int],
) -> None:
    """
    Run benchmark for a specific config (division, programming_language, query_type).
    
    Args:
        config_name: Configuration name (division, programming_language, query_type)
        benchmark_path: Path to benchmark dataset
        retrievers: Pre-computed retrievers
        queries_df: DataFrame of queries
        retriever_names: List of retriever names to evaluate
        embedding_model: Embedding model name
        k_values: K values for evaluation metrics
    """
    # Load qrels by category for this config
    category_qrels_dict = load_qrels_by_category(benchmark_path, config_name)
    
    for retriever_name in retriever_names:
        print(f"\n{'='*80}")
        print(f"Running benchmark for {config_name} with {retriever_name}")
        print(f"{'='*80}")
        
        # Get search results from retriever
        search_results = retrievers.get(retriever_name).search(queries_df, top_k=max(k_values))
        
        print(f"Evaluating {retriever_name} retriever for config: {config_name}...")
        
        # Evaluate for each category
        all_results = []
        for category, category_qrels in category_qrels_dict.items():
            # Filter search results to only include queries in this category
            category_results = {
                qid: res for qid, res in search_results.items()
                if qid in category_qrels
            }
            metrics = evaluate_retrieval(category_qrels, category_results, k_values=k_values)
            
            # Store result
            result_row = {
                'config_name': config_name,
                'category': category,
                **metrics
            }
            all_results.append(result_row)
            
            # Print summary
            print(f"  {category:12s}: MRR@10={metrics.get('mrr@10', 0):.4f}, NDCG@10={metrics.get('ndcg@10', 0):.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Display formatted results
        print(format_results_table_by_category(
            results_df,
            retriever_name=retriever_name,
            embedding_model=embedding_model
        ))
        
        # Save results with dynamic filename
        if retriever_name == 'bm25':
            output_filename = f'code_bm25_benchmark_results_by_{config_name}.csv'
        elif retriever_name == 'embedding':
            model_suffix = embedding_model.replace('/', '-').replace('\\', '-')
            output_filename = f'code_embedding_{model_suffix}_benchmark_results_by_{config_name}.csv'
        else:
            output_filename = f'code_{retriever_name}_benchmark_results_by_{config_name}.csv'
        
        output_path = Path('results') / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path.absolute()}")
    

def run_benchmark_overall(
    benchmark_path: str,
    retrievers: Dict[str, Any],
    queries_df: pd.DataFrame,
    retriever_names: List[str],
    embedding_model: str,
    k_values: List[int],   
):
    """
    Run benchmark for a overall not specific configuration.
    
    Args:
        benchmark_path: Path to benchmark dataset
        retrievers: Pre-computed retrievers
        queries_df: DataFrame of queries
        retriever_names: List of retriever names to evaluate
        embedding_model: Embedding model name
        k_values: K values for evaluation metrics
    """
    # Load qrels by category for this config
    combined_qrels = load_qrels_overall(benchmark_path)

    for retriever_name in retriever_names:
        print(f"\n{'='*80}")
        print(f"Running OVERALL/HOLISTIC benchmark with {retriever_name}")
        print(f"{'='*80}")
        
        # Get search results from retriever
        search_results = retrievers.get(retriever_name).search(queries_df, top_k=max(k_values))
        
        print(f"Evaluating {retriever_name} retriever for overall benchmark...")
        
        # Evaluate on combined qrels
        metrics = evaluate_retrieval(combined_qrels, search_results, k_values=k_values)
        
        # Print summary
        print(f"\nOverall Results:")
        for k in k_values:
            print(f"  MRR@{k}={metrics.get(f'mrr@{k}', 0):.4f}, "
                  f"NDCG@{k}={metrics.get(f'ndcg@{k}', 0):.4f}, "
                  f"Recall@{k}={metrics.get(f'recall@{k}', 0):.4f}")
        
        # Create results DataFrame
        result_row = {
            'config_name': 'overall',
            'category': 'holistic',
            **metrics
        }
        results_df = pd.DataFrame([result_row])
        
        # Save results with dynamic filename
        if retriever_name == 'bm25':
            output_filename = f'code_bm25_benchmark_results_overall.csv'
        elif retriever_name == 'embedding':
            model_suffix = embedding_model.replace('/', '-').replace('\\', '-')
            output_filename = f'code_embedding_{model_suffix}_benchmark_results_overall.csv'
        else:
            output_filename = f'code_{retriever_name}_benchmark_results_overall.csv'
        
        output_path = Path('results') / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path.absolute()}")

  

def precompute_retrievers(
    corpus: Any,
    queries: Any,
    retriever_names: List[str],
    embedding_model: str,
    batch_size: int = 16
) -> Dict[str, Any]:
    retrievers = {}

    for retriever_name in retriever_names:
        if retriever_name == 'bm25':
            retriever = BM25Retriever()
        elif retriever_name == 'embedding':
            retriever = EmbeddingRetriever(model_name=embedding_model, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown retriever: {retriever_name}")

        retriever.index(corpus_df=corpus.to_pandas())
        # check if index_queries method exists
        if hasattr(retriever, 'index_queries'):
            retriever.index_queries(queries_df=queries.to_pandas())
        retrievers[retriever_name] = retriever
    return retrievers

def main():
    args = get_command_arguments()

    # Get the benchmark dataset
    corpus, queries = get_benchmark_dataset(args.benchamark_path)

    # Index corpus once per retriever
    retrievers = precompute_retrievers(corpus, 
                                       queries, 
                                       args.retrievers, 
                                       args.embedding_model, 
                                       batch_size=args.batch_size)
    
    k_values = [int(k) for k in args.ks]
    queries_df = queries.to_pandas()

    # Run benchmarks for selected configs
    if args.by_division:
        run_benchmark_by_config(
            config_name="division",
            benchmark_path=args.benchamark_path,
            retrievers=retrievers,
            queries_df=queries_df,
            retriever_names=args.retrievers,
            embedding_model=args.embedding_model,
            k_values=k_values
        )

    if args.by_language:
        run_benchmark_by_config(
            config_name="programming_language",
            benchmark_path=args.benchamark_path,
            retrievers=retrievers,
            queries_df=queries_df,
            retriever_names=args.retrievers,
            embedding_model=args.embedding_model,
            k_values=k_values
        )

    if args.by_query_type:
        run_benchmark_by_config(
            config_name="query_type",
            benchmark_path=args.benchamark_path,
            retrievers=retrievers,
            queries_df=queries_df,
            retriever_names=args.retrievers,
            embedding_model=args.embedding_model,
            k_values=k_values
        )

    run_benchmark_overall(
        benchmark_path=args.benchamark_path,
            retrievers=retrievers,
            queries_df=queries_df,
            retriever_names=args.retrievers,
            embedding_model=args.embedding_model,
            k_values=k_values
    )


    

if __name__ == "__main__":
    main()
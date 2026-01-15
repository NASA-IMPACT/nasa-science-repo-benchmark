"""Evaluation metrics for retrieval benchmarking"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_recall_at_k(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Calculate Recall@K for retrieval results.

    Recall@K = (# relevant docs in top-K) / (# total relevant docs)

    Args:
        qrels: Ground truth relevance {query_id: {corpus_id: score}}
        results: Retrieval results {query_id: [(corpus_id, score), ...]}
        k_values: List of K values to evaluate

    Returns:
        Dict with recall@k metrics
    """
    recall_scores = {f'recall@{k}': [] for k in k_values}

    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            # No results for this query, recall is 0
            for k in k_values:
                recall_scores[f'recall@{k}'].append(0.0)
            continue

        retrieved = results[query_id]
        num_relevant = len(relevant_docs)

        if num_relevant == 0:
            continue

        # Get top-K retrieved corpus IDs for each K
        for k in k_values:
            top_k_ids = [corpus_id for corpus_id, _ in retrieved[:k]]
            num_relevant_in_k = sum(1 for cid in top_k_ids if cid in relevant_docs)
            recall_at_k = num_relevant_in_k / num_relevant
            recall_scores[f'recall@{k}'].append(recall_at_k)

    # Average across queries
    metrics = {}
    for k in k_values:
        key = f'recall@{k}'
        if recall_scores[key]:
            metrics[key] = np.mean(recall_scores[key])
        else:
            metrics[key] = 0.0

    return metrics


def calculate_ndcg_at_k(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    Args:
        qrels: Ground truth relevance {query_id: {corpus_id: score}}
        results: Retrieval results {query_id: [(corpus_id, score), ...]}
        k_values: List of K values to evaluate

    Returns:
        Dict with ndcg@k metrics
    """
    ndcg_scores = {f'ndcg@{k}': [] for k in k_values}

    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            # No results for this query, NDCG is 0
            for k in k_values:
                ndcg_scores[f'ndcg@{k}'].append(0.0)
            continue

        retrieved = results[query_id]

        for k in k_values:
            top_k = retrieved[:k]

            # Calculate DCG
            dcg = 0.0
            for i, (corpus_id, _) in enumerate(top_k):
                relevance = relevant_docs.get(corpus_id, 0)
                # DCG formula: rel / log2(i+2) where i starts at 0
                dcg += relevance / np.log2(i + 2)

            # Calculate IDCG (Ideal DCG)
            ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
            idcg = 0.0
            for i, rel in enumerate(ideal_relevances):
                idcg += rel / np.log2(i + 2)

            # NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0

            ndcg_scores[f'ndcg@{k}'].append(ndcg)

    # Average across queries
    metrics = {}
    for k in k_values:
        key = f'ndcg@{k}'
        if ndcg_scores[key]:
            metrics[key] = np.mean(ndcg_scores[key])
        else:
            metrics[key] = 0.0

    return metrics


def calculate_mrr_at_k(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calculate MRR@K (Mean Reciprocal Rank).

    MRR@K = Average of (1 / rank of first relevant doc) within top-K
    If no relevant doc in top-K, reciprocal rank is 0.

    Args:
        qrels: Ground truth relevance {query_id: {corpus_id: score}}
        results: Retrieval results {query_id: [(corpus_id, score), ...]}
        k_values: List of K values to evaluate

    Returns:
        Dict with mrr@k metrics
    """
    mrr_scores = {f'mrr@{k}': [] for k in k_values}

    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            # No results for this query, MRR is 0
            for k in k_values:
                mrr_scores[f'mrr@{k}'].append(0.0)
            continue

        retrieved = results[query_id]

        for k in k_values:
            top_k = retrieved[:k]

            # Find rank of first relevant document
            reciprocal_rank = 0.0
            for rank, (corpus_id, _) in enumerate(top_k, start=1):
                if corpus_id in relevant_docs:
                    reciprocal_rank = 1.0 / rank
                    break

            mrr_scores[f'mrr@{k}'].append(reciprocal_rank)

    # Average across queries
    metrics = {}
    for k in k_values:
        key = f'mrr@{k}'
        if mrr_scores[key]:
            metrics[key] = np.mean(mrr_scores[key])
        else:
            metrics[key] = 0.0

    return metrics


def evaluate_retrieval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Evaluate retrieval results with multiple metrics.

    Args:
        qrels: Ground truth relevance {query_id: {corpus_id: score}}
        results: Retrieval results {query_id: [(corpus_id, score), ...]}
        k_values: List of K values to evaluate

    Returns:
        Dict with all metrics (mrr@k, recall@k, and ndcg@k)
    """
    metrics = {}

    # Calculate MRR@K
    mrr_metrics = calculate_mrr_at_k(qrels, results, k_values)
    metrics.update(mrr_metrics)

    # Calculate Recall@K
    recall_metrics = calculate_recall_at_k(qrels, results, k_values)
    metrics.update(recall_metrics)

    # Calculate NDCG@K
    ndcg_metrics = calculate_ndcg_at_k(qrels, results, k_values)
    metrics.update(ndcg_metrics)

    return metrics

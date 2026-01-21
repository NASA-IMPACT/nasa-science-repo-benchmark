"""Evaluation metrics for retrieval benchmarking"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


def evaluate_retrieval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Evaluate retrieval results with multiple metrics (optimized with progress).

    Args:
        qrels: Ground truth relevance {query_id: {corpus_id: score}}
        results: Retrieval results {query_id: [(corpus_id, score), ...]}
        k_values: List of K values to evaluate

    Returns:
        Dict with all metrics (mrr@k, recall@k, and ndcg@k)
    """
    max_k = max(k_values)
    
    # Pre-compute for all queries
    recall_scores = {f'recall@{k}': [] for k in k_values}
    ndcg_scores = {f'ndcg@{k}': [] for k in k_values}
    mrr_scores = {f'mrr@{k}': [] for k in k_values}
    
    # Pre-compute log2 values for efficiency
    log2_cache = np.log2(np.arange(2, max_k + 2))
    
    for query_id, relevant_docs in tqdm(qrels.items(), desc="Evaluating queries", leave=False):
        if query_id not in results or not results[query_id]:
            # No results for this query
            for k in k_values:
                recall_scores[f'recall@{k}'].append(0.0)
                ndcg_scores[f'ndcg@{k}'].append(0.0)
                mrr_scores[f'mrr@{k}'].append(0.0)
            continue
        
        num_relevant = len(relevant_docs)
        if num_relevant == 0:
            continue
        
        retrieved = results[query_id][:max_k]  # Only process up to max_k
        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        
        # Pre-compute relevance array for retrieved docs
        relevances = np.array([relevant_docs.get(doc_id, 0) for doc_id in retrieved_ids])
        is_relevant = relevances > 0
        
        # Pre-compute ideal DCG
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)
        
        # Calculate metrics for each k
        for k in k_values:
            k_idx = min(k, len(retrieved_ids))
            
            # Recall@k
            num_relevant_in_k = np.sum(is_relevant[:k_idx])
            recall_scores[f'recall@{k}'].append(num_relevant_in_k / num_relevant)
            
            # MRR@k - find first relevant position
            relevant_positions = np.where(is_relevant[:k_idx])[0]
            if len(relevant_positions) > 0:
                mrr_scores[f'mrr@{k}'].append(1.0 / (relevant_positions[0] + 1))
            else:
                mrr_scores[f'mrr@{k}'].append(0.0)
            
            # NDCG@k
            if k_idx > 0:
                dcg = np.sum(relevances[:k_idx] / log2_cache[:k_idx])
                idcg_k = min(k, len(ideal_relevances))
                if idcg_k > 0:
                    idcg = np.sum(np.array(ideal_relevances[:idcg_k]) / log2_cache[:idcg_k])
                    ndcg_scores[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0.0)
                else:
                    ndcg_scores[f'ndcg@{k}'].append(0.0)
            else:
                ndcg_scores[f'ndcg@{k}'].append(0.0)
    
    # Aggregate metrics
    metrics = {}
    for k in k_values:
        metrics[f'recall@{k}'] = np.mean(recall_scores[f'recall@{k}']) if recall_scores[f'recall@{k}'] else 0.0
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores[f'ndcg@{k}']) if ndcg_scores[f'ndcg@{k}'] else 0.0
        metrics[f'mrr@{k}'] = np.mean(mrr_scores[f'mrr@{k}']) if mrr_scores[f'mrr@{k}'] else 0.0
    
    return metrics
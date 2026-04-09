"""Hybrid retriever combining BM25 + embedding via Reciprocal Rank Fusion (RRF)"""

import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever


class HybridRRFRetriever(BaseRetriever):
    """Hybrid retrieval using Reciprocal Rank Fusion of BM25 and dense embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', rrf_k: int = 60):
        """
        Initialize hybrid retriever.

        Args:
            model_name: Name of sentence-transformers model for dense retrieval
            rrf_k: RRF constant (default 60, standard value from Cormack et al. 2009)
        """
        self.bm25 = BM25Retriever()
        self.embedding = EmbeddingRetriever(model_name=model_name)
        self.rrf_k = rrf_k

    def index(self, corpus_df: pd.DataFrame, text_column: str = 'text') -> None:
        """Index corpus with both BM25 and embedding retrievers."""
        self.bm25.index(corpus_df, text_column)
        self.embedding.index(corpus_df, text_column)

    def search(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search using both retrievers and fuse results via RRF.

        RRF score for document d = sum(1 / (k + rank_i(d))) across retrievers.

        Args:
            queries_df: DataFrame with '_id' and 'text' columns
            top_k: Number of top results to return

        Returns:
            Dict mapping query_id to list of (corpus_id, score) tuples
        """
        print("Running hybrid search (BM25 + Embedding with RRF)...")

        bm25_results = self.bm25.search(queries_df, top_k=top_k)
        emb_results = self.embedding.search(queries_df, top_k=top_k)

        fused_results = {}
        for qid in bm25_results:
            scores = defaultdict(float)

            for rank, (cid, _) in enumerate(bm25_results[qid]):
                scores[cid] += 1.0 / (self.rrf_k + rank + 1)

            for rank, (cid, _) in enumerate(emb_results.get(qid, [])):
                scores[cid] += 1.0 / (self.rrf_k + rank + 1)

            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results[qid] = sorted_results

        print("Hybrid search complete")
        return fused_results

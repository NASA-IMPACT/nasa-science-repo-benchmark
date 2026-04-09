"""Hybrid retriever with cross-encoder reranking: BM25 + embedding candidates → reranker"""

import pandas as pd
from typing import Dict, List, Tuple
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever
from sentence_transformers import CrossEncoder


class HybridRerankerRetriever(BaseRetriever):
    """Hybrid retrieval: BM25 + embedding candidates fused and reranked by a cross-encoder."""

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        candidate_k: int = 100,
    ):
        """
        Args:
            model_name: Sentence-transformers model for dense retrieval stage
            reranker_model: Cross-encoder model for reranking
            candidate_k: Number of candidates to retrieve from each retriever before reranking
        """
        self.bm25 = BM25Retriever()
        self.embedding = EmbeddingRetriever(model_name=model_name)
        print(f"Loading cross-encoder reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        self.candidate_k = candidate_k
        self.corpus_texts = None
        self.corpus_id_to_text = None

    def index(self, corpus_df: pd.DataFrame, text_column: str = 'text') -> None:
        """Index corpus with both retrievers and store texts for reranking."""
        self.bm25.index(corpus_df, text_column)
        self.embedding.index(corpus_df, text_column)
        self.corpus_id_to_text = dict(
            zip(corpus_df['_id'].astype(str), corpus_df[text_column].fillna('').astype(str))
        )

    def search(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve candidates from BM25 + embedding, merge, then rerank with cross-encoder.

        Args:
            queries_df: DataFrame with '_id' and 'text' columns
            top_k: Number of final results to return per query

        Returns:
            Dict mapping query_id to list of (corpus_id, score) tuples
        """
        print(f"Running hybrid+reranker search (candidates per retriever: {self.candidate_k})...")

        bm25_results = self.bm25.search(queries_df, top_k=self.candidate_k)
        emb_results = self.embedding.search(queries_df, top_k=self.candidate_k)

        # Build query text lookup
        query_texts = dict(
            zip(queries_df['_id'].astype(str), queries_df['text'].fillna('').astype(str))
        )

        reranked_results = {}
        total = len(bm25_results)
        for i, qid in enumerate(bm25_results):
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Reranking query {i + 1}/{total}...")

            # Merge candidate sets (union, deduplicated)
            candidate_ids = set()
            for cid, _ in bm25_results.get(qid, []):
                candidate_ids.add(cid)
            for cid, _ in emb_results.get(qid, []):
                candidate_ids.add(cid)

            candidate_ids = list(candidate_ids)
            query_text = query_texts.get(qid, '')

            # Build (query, document) pairs for cross-encoder
            pairs = [
                (query_text, self.corpus_id_to_text.get(cid, ''))
                for cid in candidate_ids
            ]

            # Score with cross-encoder
            scores = self.reranker.predict(pairs)

            # Sort by reranker score
            scored = list(zip(candidate_ids, [float(s) for s in scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked_results[qid] = scored[:top_k]

        print("Hybrid+reranker search complete")
        return reranked_results

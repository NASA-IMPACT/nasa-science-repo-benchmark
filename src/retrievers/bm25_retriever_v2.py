"""BM25 retriever using bm25s (fast, pure Python)"""

import pandas as pd
import bm25s
from typing import Dict, List, Tuple
from .base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25-based retrieval using bm25s."""

    def __init__(self):
        self.retriever = None
        self.corpus_ids = None

    def index(self, corpus_df: pd.DataFrame, text_column: str = 'text') -> None:
        print(f"Indexing {len(corpus_df)} documents with bm25s...")

        self.corpus_ids = corpus_df['_id'].tolist()
        texts = corpus_df[text_column].fillna('').astype(str).tolist()

        # Tokenize
        corpus_tokens = bm25s.tokenize(texts, stopwords="en")
        
        # Build index
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

        print("bm25s indexing complete")

    def search(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        if self.retriever is None:
            raise ValueError("Must call index() before search()")

        print(f"Searching {len(queries_df)} queries with bm25s...")

        query_texts = queries_df['text'].fillna('').astype(str).tolist()
        query_ids = queries_df['_id'].astype(str).tolist()
        
        # Tokenize queries
        query_tokens = bm25s.tokenize(query_texts, stopwords="en")
        
        # Batch search (fast)
        results_idx, scores = self.retriever.retrieve(query_tokens, k=top_k)

        # Convert to expected format
        results = {}
        for qid, doc_indices, doc_scores in zip(query_ids, results_idx, scores):
            results[qid] = [
                (str(self.corpus_ids[idx]), float(score))
                for idx, score in zip(doc_indices, doc_scores)
            ]

        print("bm25s search complete")
        return results
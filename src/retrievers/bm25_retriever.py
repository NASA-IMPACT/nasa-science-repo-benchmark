"""BM25 retriever implementation"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from .base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25-based retrieval."""

    def __init__(self):
        self.bm25 = None
        self.corpus_ids = None
        self.tokenized_corpus = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return str(text).lower().split()

    def index(self, corpus_df: pd.DataFrame, text_column: str = 'text') -> None:
        """
        Index corpus with BM25.

        Args:
            corpus_df: DataFrame with '_id' and text_column
            text_column: Column containing text to index
        """
        print(f"Indexing {len(corpus_df)} documents with BM25...")

        self.corpus_ids = corpus_df['_id'].tolist()
        texts = corpus_df[text_column].fillna('').astype(str).tolist()

        # Tokenize all documents
        self.tokenized_corpus = [self._tokenize(text) for text in texts]

        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("BM25 indexing complete")

    def search(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search using BM25.

        Args:
            queries_df: DataFrame with '_id' and 'text' columns
            top_k: Number of top results to return

        Returns:
            Dict mapping query_id to list of (corpus_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Must call index() before search()")

        print(f"Searching {len(queries_df)} queries with BM25...")

        results = {}

        for _, row in queries_df.iterrows():
            query_id = str(row['_id'])
            query_text = str(row.get('text', ''))

            # Tokenize query
            tokenized_query = self._tokenize(query_text)

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Create results list
            results[query_id] = [
                (str(self.corpus_ids[idx]), float(scores[idx]))
                for idx in top_indices
            ]

        print(f"BM25 search complete")
        return results

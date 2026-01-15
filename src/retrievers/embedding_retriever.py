"""Embedding-based retriever implementation"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base_retriever import BaseRetriever


class EmbeddingRetriever(BaseRetriever):
    """Dense embedding-based retrieval using sentence transformers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding retriever.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_ids = None
        self.corpus_embeddings = None

    def index(self, corpus_df: pd.DataFrame, text_column: str = 'text') -> None:
        """
        Index corpus by encoding to embeddings.

        Args:
            corpus_df: DataFrame with '_id' and text_column
            text_column: Column containing text to encode
        """
        print(f"Encoding {len(corpus_df)} documents...")

        self.corpus_ids = corpus_df['_id'].tolist()
        texts = corpus_df[text_column].fillna('').astype(str).tolist()

        # Encode all documents
        self.corpus_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Indexed {len(self.corpus_ids)} documents (embedding dim: {self.corpus_embeddings.shape[1]})")

    def search(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search using embedding similarity.

        Args:
            queries_df: DataFrame with '_id' and 'text' columns
            top_k: Number of top results to return

        Returns:
            Dict mapping query_id to list of (corpus_id, score) tuples
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call index() before search()")

        print(f"Searching {len(queries_df)} queries with embeddings...")

        query_ids = queries_df['_id'].tolist()
        query_texts = queries_df['text'].fillna('').astype(str).tolist()

        # Encode queries
        query_embeddings = self.model.encode(
            query_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Calculate cosine similarities
        print("Computing similarities...")
        similarities = cosine_similarity(query_embeddings, self.corpus_embeddings)

        # Get top-k for each query
        results = {}
        for i, query_id in enumerate(query_ids):
            query_scores = similarities[i]
            top_indices = np.argsort(query_scores)[::-1][:top_k]

            results[str(query_id)] = [
                (str(self.corpus_ids[idx]), float(query_scores[idx]))
                for idx in top_indices
            ]

        print("Embedding search complete")
        return results

"""Embedding-based retriever implementation"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from .base_retriever import BaseRetriever


class EmbeddingRetriever(BaseRetriever):
    """Dense embedding-based retrieval using sentence transformers."""

    def __init__(self, model_name: str = 'nasa-impact/indus-sde-st-v0.2', use_multi_gpu: bool = True, batch_size: int = 16):
        """
        Initialize embedding retriever.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name, 
            token=os.getenv("HUGGINGFACE_TOKEN"), 
            trust_remote_code=True,
            model_kwargs={
                'torch_dtype': torch.float16,
                # 'device_map': 'cuda',  # or 'cuda:0' for specific GPU
                # 'low_cpu_mem_usage': True
            }
        )
        self.corpus_ids = None
        self.corpus_embeddings = None
        self.query_embeddings = None
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size

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
        if self.use_multi_gpu:
            pool = self.model.start_multi_process_pool()
            self.corpus_embeddings = self.model.encode_multi_process(
                texts,
                pool=pool,
                batch_size=self.batch_size,  # Adjust based on GPU memory
                show_progress_bar=True,
                chunk_size=1000  # Auto-calculate
            )
            self.model.stop_multi_process_pool(pool)
        else:
            self.corpus_embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        print(f"Indexed {len(self.corpus_ids)} documents (embedding dim: {self.corpus_embeddings.shape[1]})")


    def index_queries(self, queries_df: pd.DataFrame, text_column: str = 'text') -> None:
        """
        Pre-encode queries to embeddings for faster search.
        
        Args:
            queries_df: DataFrame with '_id' and text_column
            text_column: Column containing query text
            use_multi_gpu: Whether to use multi-GPU encoding
        """
        print(f"Pre-encoding {len(queries_df)} queries...")
        
        self.query_ids = queries_df['_id'].tolist()
        query_texts = queries_df[text_column].fillna('').astype(str).tolist()
        
        # Encode queries
        if self.use_multi_gpu:
            pool = self.model.start_multi_process_pool()
            self.query_embeddings = self.model.encode_multi_process(
                query_texts,
                pool=pool,
                batch_size=self.batch_size,
                show_progress_bar=True,
                chunk_size=1000
            )
            self.model.stop_multi_process_pool(pool)
        else:
            self.query_embeddings = self.model.encode(
                query_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        print(f"Pre-encoded {len(self.query_ids)} queries (embedding dim: {self.query_embeddings.shape[1]})")
        
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
        if self.query_embeddings is not None and len(self.query_embeddings) == len(queries_df):
            pass
        else:
            self.index_queries(queries_df)

        # Calculate cosine similarities
        print("Computing similarities...")
        similarities = cosine_similarity(self.query_embeddings, self.corpus_embeddings)

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

"""Base retriever interface"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple


class BaseRetriever(ABC):
    """Abstract base class for retrieval methods."""

    @abstractmethod
    def index(self, corpus_df: pd.DataFrame, text_column: str = 'text') -> None:
        """
        Index the corpus for retrieval.

        Args:
            corpus_df: DataFrame with '_id' and text_column
            text_column: Name of the column containing text
        """
        pass

    @abstractmethod
    def search(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search for relevant documents for each query.

        Args:
            queries_df: DataFrame with '_id' and 'text' columns
            top_k: Number of top results to return per query

        Returns:
            Dict mapping query_id to list of (corpus_id, score) tuples
        """
        pass

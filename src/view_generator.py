"""Text view generator for creating different corpus representations"""

import pandas as pd
from typing import List


class ViewGenerator:
    """Generate different text views from enriched corpus for benchmarking."""

    AVAILABLE_VIEWS = [
        'readme',
        'readme_cleaned',
        'readme_and_topics',
        'readme_and_additional_context',
        'full'
    ]

    def create_text_view(self, corpus_df: pd.DataFrame, view_name: str) -> pd.DataFrame:
        """
        Create a text view of the corpus for retrieval.

        Args:
            corpus_df: Enriched corpus DataFrame
            view_name: Name of the view to create

        Returns:
            DataFrame with '_id' and 'text' columns
        """
        if view_name not in self.AVAILABLE_VIEWS:
            raise ValueError(
                f"Unknown view '{view_name}'. Available views: {self.AVAILABLE_VIEWS}"
            )

        print(f"Creating '{view_name}' view...")

        # Create a copy with _id and text columns
        result = corpus_df[['_id']].copy()

        if view_name == 'readme':
            result['text'] = self._create_readme_view(corpus_df)
        elif view_name == 'readme_cleaned':
            result['text'] = self._create_readme_cleaned_view(corpus_df)
        elif view_name == 'readme_and_topics':
            result['text'] = self._create_readme_and_topics_view(corpus_df)
        elif view_name == 'readme_and_additional_context':
            result['text'] = self._create_readme_and_context_view(corpus_df)
        elif view_name == 'full':
            result['text'] = self._create_full_view(corpus_df)

        # Remove rows with empty text
        result = result[result['text'].str.strip().str.len() > 0]

        print(f"Created {len(result)} documents for '{view_name}' view")
        return result

    def _create_readme_view(self, df: pd.DataFrame) -> pd.Series:
        """Create view with just README content."""
        return df['readme'].fillna('').astype(str)

    def _create_readme_cleaned_view(self, df: pd.DataFrame) -> pd.Series:
        """Create view with just cleaned README content."""
        return df['readme_cleaned'].fillna('').astype(str)

    def _create_readme_and_topics_view(self, df: pd.DataFrame) -> pd.Series:
        """Create view with README + topics."""
        texts = []
        for _, row in df.iterrows():
            readme = str(row.get('readme', '')) if pd.notna(row.get('readme')) else ''

            # Format topics
            topics = row.get('topics', [])
            if isinstance(topics, list) and topics:
                topics_str = ', '.join(str(t) for t in topics)
                text = f"{readme}\n\n### Topics\n{topics_str}"
            else:
                text = readme

            texts.append(text)

        return pd.Series(texts, index=df.index)

    def _create_readme_and_context_view(self, df: pd.DataFrame) -> pd.Series:
        """Create view with README + additional context."""
        texts = []
        for _, row in df.iterrows():
            readme = str(row.get('readme', '')) if pd.notna(row.get('readme')) else ''
            context = str(row.get('additional_context', '')) if pd.notna(row.get('additional_context')) else ''

            if context:
                text = f"{readme}\n\n### Additional Context\n{context}"
            else:
                text = readme

            texts.append(text)

        return pd.Series(texts, index=df.index)

    def _create_full_view(self, df: pd.DataFrame) -> pd.Series:
        """Create view with all fields combined."""
        texts = []
        for _, row in df.iterrows():
            parts = []

            # README
            readme = str(row.get('readme', '')) if pd.notna(row.get('readme')) else ''
            if readme:
                parts.append(readme)

            # Topics
            topics = row.get('topics', [])
            if isinstance(topics, list) and topics:
                topics_str = ', '.join(str(t) for t in topics)
                parts.append(f"### Topics\n{topics_str}")

            # Additional context
            context = str(row.get('additional_context', '')) if pd.notna(row.get('additional_context')) else ''
            if context:
                parts.append(f"### Additional Context\n{context}")

            # Description
            description = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
            if description:
                parts.append(f"### Description\n{description}")

            text = '\n\n'.join(parts)
            texts.append(text)

        return pd.Series(texts, index=df.index)

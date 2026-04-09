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
        'full',
        'enriched',  # Matches LocalSearchTool's text creation
        'combined',  # Alias for enriched (paper terminology)
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
        elif view_name in ('enriched', 'combined'):
            result['text'] = self._create_enriched_view(corpus_df)

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

    def _create_enriched_view(self, df: pd.DataFrame) -> pd.Series:
        """
        Create view that exactly matches LocalSearchTool's text creation.

        Concatenates fields in this order (matching LocalSearchTool's context_columns):
        1. description
        2. reformulated_text (readme_cleaned)
        3. key_topics (topics)
        4. relevant_content (additional_context)

        Fields are concatenated with spaces, no headers.
        """
        texts = []
        for _, row in df.iterrows():
            parts = []

            # 1. description
            description = row.get('description', '')
            if pd.notna(description) and str(description).strip():
                parts.append(str(description))

            # 2. reformulated_text → readme_cleaned
            readme_cleaned = row.get('readme_cleaned', '')
            if pd.notna(readme_cleaned) and str(readme_cleaned).strip():
                parts.append(str(readme_cleaned))

            # 3. key_topics → topics
            topics = row.get('topics', '')
            try:
                if isinstance(topics, list) and len(topics) > 0:
                    topics_str = ' '.join(str(t) for t in topics)
                    if topics_str.strip():
                        parts.append(topics_str)
                elif topics is not None and str(topics).strip() and str(topics) != 'nan':
                    parts.append(str(topics))
            except:
                # Handle cases where topics might be problematic (empty arrays, etc)
                pass

            # 4. relevant_content → additional_context
            additional_context = row.get('additional_context', '')
            if pd.notna(additional_context) and str(additional_context).strip():
                parts.append(str(additional_context))

            # Concatenate with spaces (like LocalSearchTool does)
            text = ' '.join(parts)
            texts.append(text)

        return pd.Series(texts, index=df.index)

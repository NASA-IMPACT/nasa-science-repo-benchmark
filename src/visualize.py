"""Visualization utilities for benchmark results"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List


def create_comparison_plots(
    results_df: pd.DataFrame,
    output_dir: str = 'results/plots',
    k_values: List[int] = None
) -> None:
    """
    Create comparison plots for benchmark results.

    Args:
        results_df: Results DataFrame from run_benchmark()
        output_dir: Directory to save plots
        k_values: K values to plot
    """
    if k_values is None:
        k_values = [10, 50]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    for k in k_values:
        # Recall@K comparison
        recall_col = f'recall@{k}'
        if recall_col in results_df.columns:
            plt.figure(figsize=(14, 6))
            plot_data = results_df.pivot(index='view', columns='retriever', values=recall_col)
            plot_data.plot(kind='bar', width=0.8)
            plt.title(f'Recall@{k} Comparison Across Views and Retrievers')
            plt.xlabel('Text View')
            plt.ylabel(f'Recall@{k}')
            plt.legend(title='Retriever', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / f'recall_at_{k}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # NDCG@K comparison
        ndcg_col = f'ndcg@{k}'
        if ndcg_col in results_df.columns:
            plt.figure(figsize=(14, 6))
            plot_data = results_df.pivot(index='view', columns='retriever', values=ndcg_col)
            plot_data.plot(kind='bar', width=0.8)
            plt.title(f'NDCG@{k} Comparison Across Views and Retrievers')
            plt.xlabel('Text View')
            plt.ylabel(f'NDCG@{k}')
            plt.legend(title='Retriever', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / f'ndcg_at_{k}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Combined heatmap for Recall@10
    if 'recall@10' in results_df.columns:
        plt.figure(figsize=(10, 6))
        pivot_data = results_df.pivot(index='view', columns='retriever', values='recall@10')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Recall@10'})
        plt.title('Recall@10 Heatmap')
        plt.xlabel('Retriever')
        plt.ylabel('Text View')
        plt.tight_layout()
        plt.savefig(output_path / 'recall_at_10_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Combined heatmap for NDCG@10
    if 'ndcg@10' in results_df.columns:
        plt.figure(figsize=(10, 6))
        pivot_data = results_df.pivot(index='view', columns='retriever', values='ndcg@10')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'NDCG@10'})
        plt.title('NDCG@10 Heatmap')
        plt.xlabel('Retriever')
        plt.ylabel('Text View')
        plt.tight_layout()
        plt.savefig(output_path / 'ndcg_at_10_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to: {output_path.absolute()}")
    print(f"  - recall_at_*.png")
    print(f"  - ndcg_at_*.png")
    print(f"  - *_heatmap.png")

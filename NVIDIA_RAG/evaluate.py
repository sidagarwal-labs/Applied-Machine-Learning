"""
evaluate.py - Ranking evaluation metrics.

Implements standard IR metrics:
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - Recall@k
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)
"""

import numpy as np
import pandas as pd


def dcg_at_k(relevances, k):
    """Compute DCG@k."""
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0
    # DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    """
    Compute NDCG@k.
    relevances: list of relevance scores sorted by model's predicted rank.
    """
    dcg = dcg_at_k(relevances, k)
    # Ideal DCG: sort relevances descending
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(relevances, k, total_relevant):
    """
    Compute Recall@k.
    relevances: list of binary relevance scores sorted by predicted rank.
    total_relevant: total number of relevant documents for this query.
    """
    if total_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def reciprocal_rank(relevances):
    """
    Compute Reciprocal Rank.
    Returns 1/rank of the first relevant document.
    """
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(relevances):
    """
    Compute Average Precision for a single query.
    """
    if sum(relevances) == 0:
        return 0.0
    precisions = []
    relevant_count = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return sum(precisions) / sum(relevances)


def evaluate_ranking(df, score_col, k_values=[5, 10, 20]):
    """
    Evaluate a ranking model on a DataFrame.

    Args:
        df: DataFrame with columns: question, chunk_id, relevance, and score_col
        score_col: name of the column containing predicted scores
        k_values: list of k values for @k metrics

    Returns:
        dict of metric_name -> value
    """
    results = {f'NDCG@{k}': [] for k in k_values}
    results.update({f'Recall@{k}': [] for k in k_values})
    results['MRR'] = []
    results['MAP'] = []

    for question, group in df.groupby('question'):
        # Sort by predicted score (descending)
        sorted_group = group.sort_values(score_col, ascending=False)
        relevances = sorted_group['relevance'].tolist()
        total_relevant = sum(relevances)

        for k in k_values:
            results[f'NDCG@{k}'].append(ndcg_at_k(relevances, k))
            results[f'Recall@{k}'].append(recall_at_k(relevances, k, total_relevant))

        results['MRR'].append(reciprocal_rank(relevances))
        results['MAP'].append(average_precision(relevances))

    # Average across all queries
    return {metric: np.mean(values) for metric, values in results.items()}


def print_evaluation(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:15s}: {value:.4f}")
    print(f"{'='*50}")

"""
Shared utility functions for meta-evaluation analysis.
Reuses and extends functionality from newsgroup/report_utils.py
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import spearmanr, kendalltau
import json
import os
from pathlib import Path


def compute_npmi_per_topic(
    topics: List[List[str]],
    word_doc_freq: Dict[str, int],
    co_doc_freq: Dict[Tuple[str, str], int],
    total_documents: int,
    top_k: int = 10
) -> List[float]:
    """Compute per-topic NPMI (matching root StatEvaluator formula)."""
    eps = 1e-12
    scores: List[float] = []
    for topic in topics:
        if len(topic) < 2:
            scores.append(0.0)
            continue
        words = topic[:top_k]
        topic_scores = []
        for i in range(1, len(words)):
            for j in range(0, i):
                w2, w1 = words[i], words[j]
                d_w1 = word_doc_freq.get(w1, 0)
                if d_w1 == 0:
                    continue
                pair = (w1, w2) if w1 < w2 else (w2, w1)
                d_w1w2 = co_doc_freq.get(pair, 0)
                p_w1 = d_w1 / total_documents
                p_w2 = word_doc_freq.get(w2, 0) / total_documents
                p_w1w2 = d_w1w2 / total_documents
                if p_w1w2 == 0:
                    continue
                pmi = np.log((p_w1w2 + eps) / (p_w1 * p_w2 + eps))
                npmi = pmi / (-np.log(p_w1w2 + eps))
                topic_scores.append((npmi + 1) / 2)
        scores.append(float(np.mean(topic_scores)) if topic_scores else 0.0)
    return scores


def spearman_correlation(a: List[float], b: List[float]) -> float:
    """Compute Spearman correlation between two lists."""
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def kendall_correlation(a: List[float], b: List[float]) -> float:
    """Compute Kendall's tau correlation between two lists."""
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    tau, _ = kendalltau(a, b)
    return 0.0 if np.isnan(tau) else float(tau)


def pairwise_accuracy(ref: List[float], pred: List[float]) -> float:
    """Fraction of pairs where ordering agrees with reference."""
    n = len(ref)
    if n < 2 or n != len(pred):
        return 0.0
    total = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            ref_cmp = ref[i] - ref[j]
            pred_cmp = pred[i] - pred[j]
            if (ref_cmp == 0 and abs(pred_cmp) < 1e-12) or (ref_cmp * pred_cmp > 0):
                correct += 1
    return correct / total if total else 0.0


def compute_silhouette(doc_embeddings: np.ndarray, labels: List[int]) -> float:
    """Compute silhouette score for document embeddings and labels."""
    if len(set(labels)) <= 1:
        return 0.0
    try:
        return float(silhouette_score(doc_embeddings, labels, metric='cosine'))
    except Exception:
        return 0.0


def compute_nmi_ari(labels_true: List[int], labels_pred: List[int]) -> Tuple[float, float]:
    """Compute NMI and ARI between true and predicted labels."""
    if len(labels_true) != len(labels_pred) or len(labels_true) == 0:
        return 0.0, 0.0
    try:
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        return float(nmi), float(ari)
    except Exception:
        return 0.0, 0.0


def bootstrap_cv(values: List[float], n_boot: int = 30, random_state: int = 42) -> float:
    """Compute bootstrap coefficient of variation."""
    if not values:
        return 0.0
    rng = np.random.default_rng(random_state)
    samples = []
    arr = np.array(values)
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples.append(float(np.mean(arr[idx])))
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    return 0.0 if mean == 0 else (std / mean) * 100.0


def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Compute Fleiss' kappa for inter-rater agreement.
    
    Args:
        ratings: 2D array where rows are subjects and columns are raters
                Values should be categorical (0, 1, 2, ...)
    
    Returns:
        Fleiss' kappa value
    """
    n_subjects, n_raters = ratings.shape
    
    # Count agreements per category
    n_categories = int(np.max(ratings)) + 1
    p_j = np.zeros(n_categories)
    
    for j in range(n_categories):
        p_j[j] = np.sum(ratings == j) / (n_subjects * n_raters)
    
    # Calculate P_e (expected agreement by chance)
    p_e = np.sum(p_j ** 2)
    
    # Calculate P_o (observed agreement)
    p_o = 0.0
    for i in range(n_subjects):
        subject_ratings = ratings[i, :]
        n_agreements = 0
        for j in range(n_categories):
            count = np.sum(subject_ratings == j)
            n_agreements += count * (count - 1)
        p_o += n_agreements / (n_raters * (n_raters - 1))
    
    p_o = p_o / n_subjects
    
    # Calculate kappa
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


def kendall_w(ratings: np.ndarray) -> float:
    """
    Compute Kendall's W (coefficient of concordance).
    
    Args:
        ratings: 2D array where rows are subjects and columns are raters
    
    Returns:
        Kendall's W value
    """
    n_subjects, n_raters = ratings.shape
    
    # Calculate rank sums for each subject
    rank_sums = np.zeros(n_subjects)
    for i in range(n_subjects):
        subject_ratings = ratings[i, :]
        ranks = np.argsort(np.argsort(subject_ratings)) + 1  # Convert to ranks
        rank_sums[i] = np.sum(ranks)
    
    # Calculate mean rank sum
    mean_rank_sum = np.mean(rank_sums)
    
    # Calculate sum of squared deviations
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    
    # Calculate Kendall's W
    w = (12 * ss) / (n_raters ** 2 * (n_subjects ** 3 - n_subjects))
    
    return float(w)


def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_evaluation_results(results: Dict[str, Any], file_path: str) -> None:
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def create_summary_table(results: Dict[str, Any], title: str = "Evaluation Summary") -> pd.DataFrame:
    """Create a summary table from evaluation results."""
    if 'datasets' not in results:
        return pd.DataFrame()
    
    summary_data = []
    for dataset_name, dataset_results in results['datasets'].items():
        if 'metrics' in dataset_results:
            metrics = dataset_results['metrics']
            row = {'Dataset': dataset_name}
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    row[metric_name] = f"{metric_value:.3f}"
                else:
                    row[metric_name] = str(metric_value)
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def format_correlation_matrix(correlations: Dict[str, Dict[str, float]], 
                            title: str = "Correlation Matrix") -> str:
    """Format correlation matrix as a readable table."""
    if not correlations:
        return f"{title}\nNo correlation data available."
    
    # Get all unique keys
    all_keys = set()
    for row_data in correlations.values():
        all_keys.update(row_data.keys())
    all_keys = sorted(list(all_keys))
    
    # Create DataFrame
    df = pd.DataFrame(correlations).T
    df = df.reindex(columns=all_keys, fill_value=0.0)
    
    # Format as string
    formatted = f"{title}\n"
    formatted += "=" * len(title) + "\n"
    formatted += df.to_string(float_format='{:.3f}'.format)
    
    return formatted

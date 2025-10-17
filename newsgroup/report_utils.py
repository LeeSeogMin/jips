import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import spearmanr


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
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


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
    if len(set(labels)) <= 1:
        return 0.0
    try:
        return float(silhouette_score(doc_embeddings, labels, metric='cosine'))
    except Exception:
        return 0.0


def compute_nmi_ari(labels_true: List[int], labels_pred: List[int]) -> Tuple[float, float]:
    if len(labels_true) != len(labels_pred) or len(labels_true) == 0:
        return 0.0, 0.0
    try:
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        return float(nmi), float(ari)
    except Exception:
        return 0.0, 0.0


def bootstrap_cv(values: List[float], n_boot: int = 30, random_state: int = 42) -> float:
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



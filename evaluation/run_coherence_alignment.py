import json
import math
import os
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from NeuralEvaluator import TopicModelNeuralEvaluator
from sentence_transformers import SentenceTransformer


def load_topics_and_docs() -> Dict[str, Dict[str, List]]:
    datasets = {}
    # Expected files exist in data/
    for name in ["distinct", "similar", "more_similar"]:
        topics_pkl = os.path.join("data", f"topics_{name}.pkl")
        docs_csv = os.path.join("data", f"{name}_topic.csv")
        if not os.path.exists(topics_pkl) or not os.path.exists(docs_csv):
            continue
        topics = pd.read_pickle(topics_pkl)
        df = pd.read_csv(docs_csv)
        datasets[name] = {"topics": topics, "docs": df["text"].tolist()}
    # 20NG (optional): if preprocessed available under data/
    # Add hook here if needed
    return datasets


def load_llm_coherence() -> Dict[str, List[float]]:
    # Try pickle first
    pkl_path = os.path.join("data", "llm_evaluation_results.pkl")
    csv_path = os.path.join("data", "llm_evaluation_comparison.csv")
    if os.path.exists(pkl_path):
        llm = pd.read_pickle(pkl_path)
        # Expect structure: {dataset: {topic_index: coherence_score}}
        return {k: list(v.values()) for k, v in llm.get("coherence", {}).items()} if isinstance(llm, dict) else {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Expect columns: dataset, topic_index, llm_coherence
        out: Dict[str, List[float]] = {}
        for name, g in df.groupby("dataset"):
            out[name] = g.sort_values("topic_index")["llm_coherence"].tolist()
        return out
    return {}


def build_keyword_embedding_map(datasets: Dict[str, Dict[str, List]]) -> Dict[str, np.ndarray]:
    # Collect unique keywords across datasets
    unique_keywords: Set[str] = set()
    for bundle in datasets.values():
        for topic in bundle["topics"]:
            for kw in topic:
                unique_keywords.add(str(kw))
    if not unique_keywords:
        return {}
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_list = sorted(unique_keywords)
    embs = model.encode(kw_list, show_progress_bar=False)
    return {k: e for k, e in zip(kw_list, embs)}


def evaluate_with_params(datasets, kw_emb_map, alpha, beta, gamma, tau, m_frac, delta) -> Dict[str, float]:
    rhos: Dict[str, float] = {}
    for name, bundle in datasets.items():
        topics = bundle["topics"]
        docs = bundle["docs"]
        # Use keyword embeddings map for coherence
        evaluator = TopicModelNeuralEvaluator(model_embeddings=kw_emb_map.copy(), embedding_dim=384, device="cpu")
        evaluator.coherence_alpha = alpha
        evaluator.coherence_beta = beta
        evaluator.coherence_gamma = gamma
        evaluator.coherence_tau = tau
        evaluator.coherence_m_frac = m_frac
        evaluator.coherence_delta = delta
        # Compute per-topic coherence
        per_topic = []
        for t in topics:
            per_topic.append(evaluator._calculate_coherence_score(t))
        # Compare with LLM
        llm = llm_coherence.get(name)
        if llm and len(llm) == len(per_topic) and len(per_topic) > 1:
            rho, _ = spearmanr(per_topic, llm)
            rhos[name] = float(rho if not math.isnan(rho) else 0.0)
        else:
            rhos[name] = 0.0
    return rhos


if __name__ == "__main__":
    datasets = load_topics_and_docs()
    llm_coherence = load_llm_coherence()
    kw_emb_map = build_keyword_embedding_map(datasets)

    # Parameter grid (≤60 combos)
    alphas = [0.5, 0.65, 0.8]
    betas = [0.2, 0.25, 0.35]
    gammas = [0.1, 0.15, 0.25]
    taus = [0.1, 0.2, 0.3]
    m_fracs = [0.2, 0.3, 0.4]
    deltas = [0.1, 0.15, 0.2]

    best = {"score": -1, "params": None, "detail": None}
    results = []

    for a in alphas:
        for b in betas:
            for g in gammas:
                for t in taus:
                    for mf in m_fracs:
                        for d in deltas:
                            rhos = evaluate_with_params(datasets, kw_emb_map, a, b, g, t, mf, d)
                            # Average rho across datasets that have LLM
                            vals = [v for v in rhos.values()]
                            avg = float(np.mean(vals)) if vals else 0.0
                            entry = {"alpha": a, "beta": b, "gamma": g, "tau": t, "m_frac": mf, "delta": d, "rhos": rhos, "avg_rho": avg}
                            results.append(entry)
                            if avg > best["score"]:
                                best = {"score": avg, "params": {"alpha": a, "beta": b, "gamma": g, "tau": t, "m_frac": mf, "delta": d}, "detail": rhos}

    report = {"best": best, "results": results}
    with open("coherence_alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Best avg Spearman rho:", best["score"]) 
    print("Best params:", best["params"]) 


import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

# Use project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from newsgroup.run_20newsgroups_validation import load_20newsgroups_data, run_cte_model
from newsgroup.report_utils import (
    compute_npmi_per_topic,
    spearman_correlation,
    pairwise_accuracy,
    compute_silhouette,
    compute_nmi_ari,
    bootstrap_cv,
)
from StatEvaluator import TopicModelStatEvaluator
from NeuralEvaluator import TopicModelNeuralEvaluator


def collect_stat_scores(topics: List[List[str]], tokenized_texts: List[List[str]]) -> Dict[str, List[float]]:
    # Build frequencies
    word_doc_freq: Dict[str, int] = {}
    co_doc_freq: Dict[Tuple[str, str], int] = {}
    for doc in tokenized_texts:
        doc_words = set(doc)
        for w in doc_words:
            word_doc_freq[w] = word_doc_freq.get(w, 0) + 1
        for w1 in doc_words:
            for w2 in doc_words:
                if w1 < w2:
                    key = (w1, w2)
                    co_doc_freq[key] = co_doc_freq.get(key, 0) + 1
    total_docs = len(tokenized_texts)
    # Per-topic NPMI
    per_topic_coh = compute_npmi_per_topic(topics, word_doc_freq, co_doc_freq, total_docs)
    # Diversity per-topic proxy: unique share within topic (optional: treat same for all topics)
    per_topic_div = []
    for t in topics:
        vocab = set(t)
        per_topic_div.append(len(vocab) / len(t) if t else 0.0)
    # Distinctiveness at set-level (JSD); per-topic proxy: same value for all topics
    evaluator = TopicModelStatEvaluator()
    evaluator.set_model_stats(
        word_doc_freq=word_doc_freq,
        co_doc_freq=co_doc_freq,
        total_documents=total_docs,
    )
    jsd = evaluator._calculate_jsd(topics, list(range(total_docs)))
    per_topic_dis = [jsd] * len(topics)
    return {'coherence': per_topic_coh, 'distinctiveness': per_topic_dis, 'diversity': per_topic_div}


def collect_semantic_scores(topics: List[List[str]], tokenized_texts: List[List[str]]) -> Dict[str, List[float]]:
    # Build docs
    docs = [' '.join(t) for t in tokenized_texts]
    # Dummy assignments by max overlap to ensure lengths match
    topic_sets = [set(t) for t in topics]
    assignments = []
    for tokens in tokenized_texts:
        s = set(tokens)
        overlaps = [len(s & ts) for ts in topic_sets]
        assignments.append(int(np.argmax(overlaps)) if overlaps else 0)
    evaluator = TopicModelNeuralEvaluator(device='cpu')
    evaluator.use_dynamic_embeddings = True
    res = evaluator.evaluate(topics, docs, assignments)
    det = res.get('detailed_scores', {})
    coh = det.get('coherence', {}).get('topic_coherence', [])
    # Approximate per-topic distinctiveness by pairwise averaging: use matrix if available
    dis_info = det.get('distinctiveness', {})
    per_topic_dis: List[float] = []
    if 'topic_pairs' in dis_info:
        n = len(topics)
        pair = dis_info['topic_pairs']
        for i in range(n):
            vals = []
            for j in range(n):
                if i == j:
                    continue
                key = f'pair_{min(i,j)}_{max(i,j)}'
                if key in pair:
                    vals.append(pair[key])
            per_topic_dis.append(float(np.mean(vals)) if vals else 0.0)
    else:
        per_topic_dis = [res.get('distinctiveness', 0.0)] * len(topics)
    # Diversity per-topic proxy matching stat approach
    per_topic_div = []
    for t in topics:
        vocab = set(t)
        per_topic_div.append(len(vocab) / len(t) if t else 0.0)
    return {'coherence': coh, 'distinctiveness': per_topic_dis, 'diversity': per_topic_div}


def collect_llm_coherence_multi(topics: List[List[str]]) -> Dict[str, List[float]]:
    """Return per-provider coherence arrays for available LLMs."""
    providers: Dict[str, List[float]] = {}

    def eval_provider(name: str, ctor) -> None:
        try:
            ev = ctor()
            vals: List[float] = []
            for t in topics:
                score, _ = ev.evaluate_coherence(t[:8])
                vals.append(score)
            if vals:
                providers[name] = vals
        except Exception:
            pass

    # Try Anthropic, OpenAI, Grok; keep Gemini optional and skipped by default due to instability
    try:
        from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM as Anth
        eval_provider('Claude', Anth)
    except Exception:
        pass
    try:
        from topic_llm.openai_topic_evaluator import TopicEvaluatorLLM as OAI
        eval_provider('OpenAI', OAI)
    except Exception:
        pass
    try:
        from topic_llm.grok_topic_evaluator import TopicEvaluatorLLM as Grok
        eval_provider('Grok', Grok)
    except Exception:
        pass
    # Gemini intentionally excluded from reporting due to instability in this study

    return providers


def main():
    print("\n=== 20NG Metric Validation ===")
    df = load_20newsgroups_data(n_samples=1000, random_state=42)
    texts = df['text'].tolist()
    topics, tokenized_texts, cte = run_cte_model(texts, n_topics=5, n_keywords=10)

    # Collect per-topic scores
    stat = collect_stat_scores(topics, tokenized_texts)
    sem = collect_semantic_scores(topics, tokenized_texts)
    llm_all = collect_llm_coherence_multi(topics)

    # LLM alignment (coherence)
    if llm_all:
        print("\n-- LLM Alignment by Provider (Coherence) --")
        rows = []
        for name, coh in llm_all.items():
            rho_stat = spearman_correlation(coh, stat['coherence'])
            rho_sem = spearman_correlation(coh, sem['coherence'])
            pw_stat = pairwise_accuracy(coh, stat['coherence'])
            pw_sem = pairwise_accuracy(coh, sem['coherence'])
            rows.append({'LLM': name, 'Spearman_Stat': rho_stat, 'Spearman_Sem': rho_sem, 'PW_Stat': pw_stat, 'PW_Sem': pw_sem, 'LLM_AvgCoh': float(np.mean(coh))})
        df_llm = pd.DataFrame(rows)
        print(df_llm.to_string(index=False, formatters={k: (lambda v: f"{v:.3f}") for k in df_llm.columns if k != 'LLM'}))
    else:
        print("\n[WARN] No LLM coherence available (API issues). Skipping alignment block.")

    # Label-based separation (using aggregated labels from loader)
    agg_labels = df['agg_category'].astype('category').cat.codes.values.tolist()
    # Build doc embeddings for silhouette
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(texts, show_progress_bar=False)
    # Topic assignments via CTE if available
    if hasattr(cte, 'get_document_topics'):
        pred = cte.get_document_topics(cte.get_word_embeddings())
    elif hasattr(cte, 'cluster_labels'):
        pred = list(getattr(cte, 'cluster_labels'))
    else:
        # Overlap-based
        topic_sets = [set(t) for t in topics]
        pred = []
        for tokens in tokenized_texts:
            s = set(tokens)
            overlaps = [len(s & ts) for ts in topic_sets]
            pred.append(int(np.argmax(overlaps)) if overlaps else 0)
    sil = compute_silhouette(doc_embeddings, pred)
    nmi, ari = compute_nmi_ari(agg_labels, pred)
    print("\n-- Label-based Separation --")
    print(f"Silhouette (cosine): {sil:.3f}")
    print(f"NMI: {nmi:.3f}, ARI: {ari:.3f}")

    # Stability via bootstrap (coherence vectors)
    cv_stat = bootstrap_cv(stat['coherence'])
    cv_sem = bootstrap_cv(sem['coherence'])
    print("\n-- Stability (Bootstrap CV of Coherence) --")
    print(f"CV Stat: {cv_stat:.1f}%, CV Sem: {cv_sem:.1f}%")

    # Basic evaluation results (overall scores)
    print("\n-- Basic Evaluation Results --")
    print("Statistical Method:")
    print(f"  Overall Coherence: {np.mean(stat['coherence']):.3f}")
    print(f"  Overall Distinctiveness: {np.mean(stat['distinctiveness']):.3f}")
    print(f"  Overall Diversity: {np.mean(stat['diversity']):.3f}")
    
    print("\nSemantic Method:")
    print(f"  Overall Coherence: {np.mean(sem['coherence']):.3f}")
    print(f"  Overall Distinctiveness: {np.mean(sem['distinctiveness']):.3f}")
    print(f"  Overall Diversity: {np.mean(sem['diversity']):.3f}")
    
    if llm_all:
        print("\nLLM Methods:")
        for name, coh in llm_all.items():
            print(f"  {name} Overall Coherence: {np.mean(coh):.3f}")
    
    # Summary table per-topic (first 5 shown)
    k = min(5, len(topics))
    df_summary = pd.DataFrame({
        'Stat_Coh': stat['coherence'][:k],
        'Sem_Coh': sem['coherence'][:k],
        'LLM_Coh': (list(llm_all.values())[0][:k] if llm_all else [np.nan]*k),
        'Stat_Dis': stat['distinctiveness'][:k],
        'Sem_Dis': sem['distinctiveness'][:k],
        'Stat_Div': stat['diversity'][:k],
        'Sem_Div': sem['diversity'][:k],
    }, index=[f'Topic {i+1}' for i in range(k)])
    print("\n-- Sample Per-topic Metrics --")
    print(df_summary.map(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else x).to_string())


if __name__ == '__main__':
    main()



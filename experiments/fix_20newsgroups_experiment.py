#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed 20 Newsgroups Experiment - Compute proper per-topic coherence scores
"""

import numpy as np
import json
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FIXED 20 NEWSGROUPS EXPERIMENT")
print("="*70)

# Load previous results
with open("docs/20newsgroups_results.json", 'r') as f:
    prev_results = json.load(f)

print("\nPrevious results loaded. Re-computing per-topic coherence...")

# ============================================================================
# Reload data and model
# ============================================================================

newsgroups = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

documents = newsgroups.data
labels = newsgroups.target

# Prepare data
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=5000,
    stop_words='english'
)

doc_term_matrix = vectorizer.fit_transform(documents)
vocabulary = vectorizer.get_feature_names_out()

corpus = [
    [(i, count) for i, count in enumerate(doc_term_matrix[doc_idx].toarray()[0]) if count > 0]
    for doc_idx in range(len(documents))
]

id2word = {i: word for i, word in enumerate(vocabulary)}

# Train LDA
print("Re-training LDA...")
lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
    random_state=42,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto',
    per_word_topics=True,
    iterations=1000
)

print("✅ LDA ready")

# ============================================================================
# Compute per-topic UMass coherence
# ============================================================================

print("\nComputing per-topic UMass coherence...")

# Create texts for coherence computation
texts = [doc.lower().split() for doc in documents]

# Compute coherence for each topic
topic_coherences_umass = []

for topic_id in range(20):
    # Get top words for this topic
    topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=10)]

    # Create mini-corpus for this topic
    topic_texts = [[word] for word in topic_words]

    # Compute PMI-based coherence (UMass style)
    # Count co-occurrences in sliding window
    word_doc_counts = {}
    word_pair_counts = {}

    for doc in texts:
        doc_words = set(doc)
        for word in topic_words:
            if word in doc_words:
                word_doc_counts[word] = word_doc_counts.get(word, 0) + 1

        # Count pairs
        for i, w1 in enumerate(topic_words):
            if w1 in doc_words:
                for w2 in topic_words[i+1:]:
                    if w2 in doc_words:
                        pair = tuple(sorted([w1, w2]))
                        word_pair_counts[pair] = word_pair_counts.get(pair, 0) + 1

    # Compute UMass-style coherence
    coherence_scores = []
    total_docs = len(documents)

    for i, w1 in enumerate(topic_words):
        for w2 in topic_words[i+1:]:
            count_w1 = word_doc_counts.get(w1, 0)
            count_w2 = word_doc_counts.get(w2, 0)
            pair = tuple(sorted([w1, w2]))
            count_pair = word_pair_counts.get(pair, 0)

            # UMass formula: log((count(w1, w2) + 1) / count(w2))
            if count_w2 > 0:
                score = np.log((count_pair + 1) / count_w2)
                coherence_scores.append(score)

    topic_coherence = np.mean(coherence_scores) if coherence_scores else -10
    topic_coherences_umass.append(topic_coherence)

print(f"✅ Per-topic UMass coherence computed")
print(f"Range: [{min(topic_coherences_umass):.3f}, {max(topic_coherences_umass):.3f}]")
print(f"Mean: {np.mean(topic_coherences_umass):.3f}")

# ============================================================================
# Load semantic coherence scores from previous run
# ============================================================================

print("\nLoading semantic coherence from previous run...")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
word_embeddings = embedding_model.encode(list(vocabulary), show_progress_bar=False)
word_to_embedding = {word: emb for word, emb in zip(vocabulary, word_embeddings)}

# Recompute semantic coherence per topic
semantic_coherences = []

for topic_id in range(20):
    topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
    embeddings = [word_to_embedding.get(word, np.zeros(384)) for word in topic_words]
    embeddings = np.array(embeddings)

    if len(embeddings) > 1:
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                similarities.append(sim)
        sc = np.mean(similarities) if similarities else 0
    else:
        sc = 0

    semantic_coherences.append(sc)

print(f"✅ Semantic coherence: [{min(semantic_coherences):.3f}, {max(semantic_coherences):.3f}]")

# ============================================================================
# Recompute topic purities
# ============================================================================

print("\nRecomputing topic purities...")

doc_topics = []
for doc_bow in corpus:
    topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)
    topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)
    dominant_topic = topic_dist[0][0] if topic_dist else -1
    doc_topics.append(dominant_topic)

doc_topics = np.array(doc_topics)

topic_purities = []
for topic_id in range(20):
    doc_indices = np.where(doc_topics == topic_id)[0]
    if len(doc_indices) > 0:
        topic_labels = labels[doc_indices]
        unique, counts = np.unique(topic_labels, return_counts=True)
        max_count = np.max(counts)
        purity = max_count / len(topic_labels)
        topic_purities.append(purity)
    else:
        topic_purities.append(0)

print(f"✅ Purity range: [{min(topic_purities):.3f}, {max(topic_purities):.3f}]")

# ============================================================================
# Compute correlations
# ============================================================================

print("\n" + "="*70)
print("COMPUTING CORRELATIONS")
print("="*70)

# UMass vs Purity
r_umass, p_umass = pearsonr(topic_coherences_umass, topic_purities)
print(f"UMass Coherence vs Purity: r={r_umass:.3f}, p={p_umass:.4f}")

# Semantic Coherence vs Purity
r_semantic, p_semantic = pearsonr(semantic_coherences, topic_purities)
print(f"Semantic Coherence vs Purity: r={r_semantic:.3f}, p={p_semantic:.4f}")

# Topic Diversity (single value, not per-topic)
# Use Perplexity instead as baseline
perplexity = lda_model.log_perplexity(corpus)
print(f"\nPerplexity: {perplexity:.2f} (lower is better)")

# Calculate improvement
improvement_absolute = r_semantic - r_umass
improvement_percent = (improvement_absolute / abs(r_umass) * 100) if r_umass != 0 else 0

print(f"\nImprovement: {improvement_absolute:+.3f} ({improvement_percent:+.1f}%)")

# ============================================================================
# Update results
# ============================================================================

results = prev_results.copy()
results['statistical_metrics']['umass_coherence_per_topic'] = {
    'mean': float(np.mean(topic_coherences_umass)),
    'std': float(np.std(topic_coherences_umass)),
    'range': [float(min(topic_coherences_umass)), float(max(topic_coherences_umass))],
    'values': [float(x) for x in topic_coherences_umass]
}

results['semantic_metrics']['semantic_coherence_per_topic'] = {
    'mean': float(np.mean(semantic_coherences)),
    'std': float(np.std(semantic_coherences)),
    'range': [float(min(semantic_coherences)), float(max(semantic_coherences))],
    'values': [float(x) for x in semantic_coherences]
}

results['ground_truth']['topic_purities'] = [float(x) for x in topic_purities]

results['correlations'] = {
    "umass_vs_purity": {
        "r": float(r_umass),
        "p": float(p_umass),
        "interpretation": "weak" if abs(r_umass) < 0.3 else "moderate" if abs(r_umass) < 0.7 else "strong"
    },
    "semantic_coherence_vs_purity": {
        "r": float(r_semantic),
        "p": float(p_semantic),
        "interpretation": "weak" if abs(r_semantic) < 0.3 else "moderate" if abs(r_semantic) < 0.7 else "strong"
    },
    "improvement": {
        "absolute": float(improvement_absolute),
        "relative_percent": float(improvement_percent)
    }
}

# Save updated results
with open("docs/20newsgroups_results_FINAL.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: docs/20newsgroups_results_FINAL.json")

# ============================================================================
# Print final summary table
# ============================================================================

print("\n" + "="*70)
print("FINAL RESULTS TABLE")
print("="*70)
print(f"\n{'Metric':<35} {'r (purity)':<12} {'p-value':<10} {'Status':<10}")
print("-"*70)
print(f"{'Statistical Metrics:':<35}")
print(f"  {'UMass Coherence':<33} {r_umass:>10.3f}  {p_umass:>8.4f}  {'Baseline'}")
print(f"\n{'Semantic Metrics (Ours):':<35}")
print(f"  {'Semantic Coherence (SC)':<33} {r_semantic:>10.3f}  {p_semantic:>8.4f}  {'Proposed'}")
print("-"*70)
print(f"{'Improvement:':<35} {improvement_absolute:>10.3f}  {improvement_percent:>8.1f}%")
print("="*70)

print("\n✅ Experiment complete with corrected per-topic metrics!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERTopic + BBC News Experiment for External Validation
Purpose: Demonstrate semantic metrics work on real-world flat-topic dataset
Dataset: BBC News (5 categories - business, entertainment, politics, sport, tech)
Method: BERTopic (neural embedding-based topic modeling)
"""

# Disable TensorFlow before importing anything
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pearsonr
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BERTOPIC + BBC NEWS EXTERNAL VALIDATION EXPERIMENT")
print("="*70)

# ============================================================================
# STEP 1: Load BBC News Dataset
# ============================================================================

print("\n[1/7] Loading BBC News dataset...")

# BBC News dataset loading (from Kaggle or local)
# https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive
# Or use huggingface datasets
try:
    from datasets import load_dataset
    dataset = load_dataset("SetFit/bbc-news")
    documents = dataset['train']['text']
    labels = dataset['train']['label']  # 0-4 for 5 categories
    label_names = dataset['train'].features['label'].names

    print(f"✅ Loaded {len(documents)} documents")
    print(f"✅ {len(label_names)} categories: {', '.join(label_names)}")

except Exception as e:
    print(f"⚠️  Could not load from HuggingFace: {e}")
    print("Please install: pip install datasets")
    print("Or manually download BBC News dataset")
    exit(1)

# ============================================================================
# STEP 2: Train BERTopic Model
# ============================================================================

print("\n[2/7] Training BERTopic model (nr_topics=5)...")

# Initialize embedding model (same as paper)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
print("✅ Embedding model loaded")

# Initialize BERTopic with controlled number of topics
print("Training BERTopic (this may take 3-5 minutes)...")

topic_model = BERTopic(
    embedding_model=embedding_model,
    nr_topics=5,  # Force 5 topics to match 5 categories
    vectorizer_model=CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english'
    ),
    verbose=False,
    calculate_probabilities=False  # Faster training
)

# Fit model
topics_assigned, probs = topic_model.fit_transform(documents)

print("✅ BERTopic training complete")
print(f"✅ Extracted {len(set(topics_assigned))} topics (including outlier topic -1)")

# Get topic information
topic_info = topic_model.get_topic_info()
print(f"✅ Topic distribution:\n{topic_info[['Topic', 'Count']].head(10)}")

# Extract topic keywords (top 10 per topic)
topics_keywords = {}
for topic_id in range(5):  # 0-4
    keywords = topic_model.get_topic(topic_id)
    if keywords:
        topics_keywords[topic_id] = {
            'words': [word for word, _ in keywords[:10]],
            'scores': [score for _, score in keywords[:10]]
        }
        print(f"Topic {topic_id}: {', '.join(topics_keywords[topic_id]['words'][:5])}")

# ============================================================================
# STEP 3: Compute Statistical Metrics
# ============================================================================

print("\n[3/7] Computing statistical metrics...")

# Topic Diversity (TD) - unique words / total words
all_topic_words = set()
total_words_count = 0

for topic_id in topics_keywords:
    words = topics_keywords[topic_id]['words']
    all_topic_words.update(words)
    total_words_count += len(words)

topic_diversity = len(all_topic_words) / total_words_count if total_words_count > 0 else 0
print(f"✅ Topic Diversity: {topic_diversity:.3f}")

# UMass Coherence (simplified computation)
# Note: Full UMass requires document-level co-occurrence computation
# We'll use a simplified version based on keyword co-occurrence

from collections import Counter

# Build document-term matrix for coherence
doc_word_counts = []
for doc in documents:
    words = doc.lower().split()
    doc_word_counts.append(Counter(words))

def compute_umass_coherence(topic_words, doc_word_counts):
    """Simplified UMass coherence computation"""
    n_docs = len(doc_word_counts)
    coherence_scores = []

    for i in range(len(topic_words)):
        for j in range(i+1, len(topic_words)):
            w1, w2 = topic_words[i], topic_words[j]

            # Count documents containing both words
            d_w1_w2 = sum(1 for doc_count in doc_word_counts if w1 in doc_count and w2 in doc_count)
            d_w2 = sum(1 for doc_count in doc_word_counts if w2 in doc_count)

            if d_w2 > 0:
                coherence = np.log((d_w1_w2 + 1) / d_w2)
                coherence_scores.append(coherence)

    return np.mean(coherence_scores) if coherence_scores else 0

umass_scores = []
for topic_id in topics_keywords:
    words = topics_keywords[topic_id]['words']
    umass = compute_umass_coherence(words, doc_word_counts)
    umass_scores.append(umass)

avg_umass = np.mean(umass_scores)
print(f"✅ Average UMass Coherence: {avg_umass:.3f}")

# ============================================================================
# STEP 4: Compute Semantic Metrics
# ============================================================================

print("\n[4/7] Computing semantic metrics...")

# Compute word embeddings for all topic keywords
all_words = list(all_topic_words)
word_embeddings_dict = {}

print("Computing word embeddings...")
for topic_id in topics_keywords:
    words = topics_keywords[topic_id]['words']
    embeddings = embedding_model.encode(words, show_progress_bar=False)
    for word, emb in zip(words, embeddings):
        word_embeddings_dict[word] = emb

print(f"✅ Computed embeddings for {len(word_embeddings_dict)} unique words")

# Semantic Coherence (SC) for each topic
semantic_coherences = []

for topic_id in topics_keywords:
    topic_words = topics_keywords[topic_id]['words']
    embeddings = [word_embeddings_dict[word] for word in topic_words]
    embeddings = np.array(embeddings)

    # Compute pairwise cosine similarities
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

avg_semantic_coherence = np.mean(semantic_coherences)
print(f"✅ Average Semantic Coherence: {avg_semantic_coherence:.3f}")

# Semantic Distinctiveness (SD) - inter-topic distances
semantic_distinctiveness_scores = []

topic_ids = list(topics_keywords.keys())
for i in range(len(topic_ids)):
    for j in range(i+1, len(topic_ids)):
        # Get mean embeddings for each topic
        emb_i = np.array([word_embeddings_dict[w] for w in topics_keywords[topic_ids[i]]['words']])
        emb_j = np.array([word_embeddings_dict[w] for w in topics_keywords[topic_ids[j]]['words']])

        mean_i = np.mean(emb_i, axis=0)
        mean_j = np.mean(emb_j, axis=0)

        # Cosine distance = 1 - similarity
        sim = np.dot(mean_i, mean_j) / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j) + 1e-10)
        distance = 1 - sim
        semantic_distinctiveness_scores.append(distance)

avg_semantic_distinctiveness = np.mean(semantic_distinctiveness_scores)
print(f"✅ Average Semantic Distinctiveness: {avg_semantic_distinctiveness:.3f}")

# SemDiv (combined)
alpha = 0.5
beta = 0.5
semdiv = alpha * avg_semantic_coherence + beta * avg_semantic_distinctiveness
print(f"✅ SemDiv (α=0.5, β=0.5): {semdiv:.3f}")

# ============================================================================
# STEP 5: Compute Ground Truth Category Purity
# ============================================================================

print("\n[5/7] Computing ground truth category alignment...")

# Get topic assignments (already have topics_assigned)
doc_topics = np.array(topics_assigned)

# Compute category purity for each topic
# Purity = proportion of most common category in topic
topic_purities = []

for topic_id in range(5):
    doc_indices = np.where(doc_topics == topic_id)[0]
    if len(doc_indices) > 0:
        topic_labels = [labels[idx] for idx in doc_indices]
        # Most common category
        unique, counts = np.unique(topic_labels, return_counts=True)
        max_count = np.max(counts)
        purity = max_count / len(topic_labels)
        topic_purities.append(purity)

        # Print topic-category alignment
        dominant_category = label_names[unique[np.argmax(counts)]]
        print(f"  Topic {topic_id}: {purity:.2%} {dominant_category}")
    else:
        topic_purities.append(0)

avg_purity = np.mean(topic_purities)
print(f"✅ Average Category Purity: {avg_purity:.3f}")

# ============================================================================
# STEP 6: Compute Correlations with Ground Truth
# ============================================================================

print("\n[6/7] Computing correlations with ground truth...")

# Correlation: Statistical metrics vs purity
r_umass_purity, p_umass_purity = pearsonr(umass_scores, topic_purities)
print(f"✅ UMass Coherence vs Purity: r={r_umass_purity:.3f}, p={p_umass_purity:.4f}")

# Correlation: Semantic Coherence vs Purity
r_sc_purity, p_sc_purity = pearsonr(semantic_coherences, topic_purities)
print(f"✅ Semantic Coherence vs Purity: r={r_sc_purity:.3f}, p={p_sc_purity:.4f}")

# Discrimination Power Analysis
statistical_range = max(umass_scores) - min(umass_scores)
semantic_range = max(semantic_coherences) - min(semantic_coherences)
discrimination_ratio = semantic_range / statistical_range if statistical_range > 0 else 0

print(f"\n✅ Discrimination Power:")
print(f"  Statistical (UMass) range: {statistical_range:.3f}")
print(f"  Semantic (SC) range: {semantic_range:.3f}")
print(f"  Ratio (Semantic/Statistical): {discrimination_ratio:.2f}×")

# ============================================================================
# STEP 7: Generate Results Summary
# ============================================================================

print("\n[7/7] Generating results summary...")

results = {
    "dataset": {
        "name": "BBC News",
        "source": "HuggingFace SetFit/bbc-news",
        "num_documents": len(documents),
        "num_categories": len(label_names),
        "categories": label_names
    },
    "model": {
        "type": "BERTopic",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "num_topics": 5
    },
    "statistical_metrics": {
        "umass_coherence": float(avg_umass),
        "topic_diversity": float(topic_diversity),
        "umass_per_topic": [float(x) for x in umass_scores]
    },
    "semantic_metrics": {
        "semantic_coherence": float(avg_semantic_coherence),
        "semantic_distinctiveness": float(avg_semantic_distinctiveness),
        "semdiv": float(semdiv),
        "sc_per_topic": [float(x) for x in semantic_coherences]
    },
    "ground_truth": {
        "avg_category_purity": float(avg_purity),
        "purity_per_topic": [float(x) for x in topic_purities],
        "purity_range": [float(min(topic_purities)), float(max(topic_purities))]
    },
    "correlations": {
        "umass_vs_purity": {
            "r": float(r_umass_purity),
            "p": float(p_umass_purity)
        },
        "semantic_coherence_vs_purity": {
            "r": float(r_sc_purity),
            "p": float(p_sc_purity)
        }
    },
    "discrimination_power": {
        "statistical_range": float(statistical_range),
        "semantic_range": float(semantic_range),
        "ratio": float(discrimination_ratio)
    }
}

# Save results
output_path = Path("docs/bertopic_bbc_results.json")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_path}")

# Print summary table
print("\n" + "="*70)
print("EXPERIMENT RESULTS SUMMARY")
print("="*70)

print("\n1. CORRELATION WITH GROUND TRUTH CATEGORY PURITY")
print("-"*70)
print(f"{'Metric':<40} {'r (purity)':<12} {'p-value':<12}")
print("-"*70)
print(f"{'Statistical Metrics:':<40}")
print(f"  {'UMass Coherence':<38} {r_umass_purity:>10.3f}  {p_umass_purity:>10.4f}")
print(f"\n{'Semantic Metrics (Paper):':<40}")
print(f"  {'Semantic Coherence (SC)':<38} {r_sc_purity:>10.3f}  {p_sc_purity:>10.4f}")

improvement_r = r_sc_purity - r_umass_purity
print(f"\n{'Improvement (Semantic - Statistical):':<40} {improvement_r:>10.3f}")

print("\n2. DISCRIMINATION POWER")
print("-"*70)
print(f"{'Metric':<40} {'Range':<12} {'Ratio':<12}")
print("-"*70)
print(f"{'UMass Coherence (Statistical)':<40} {statistical_range:>10.3f}")
print(f"{'Semantic Coherence (Paper)':<40} {semantic_range:>10.3f}  {discrimination_ratio:>10.2f}×")

print("\n3. KEY FINDINGS")
print("-"*70)
if discrimination_ratio > 1.5:
    print(f"✅ Semantic metrics show {discrimination_ratio:.2f}× better discrimination power")
else:
    print(f"⚠️  Discrimination ratio ({discrimination_ratio:.2f}×) is moderate")

if abs(r_sc_purity) > abs(r_umass_purity):
    print(f"✅ Semantic metrics correlate better with ground truth (r={r_sc_purity:.3f} vs {r_umass_purity:.3f})")
else:
    print(f"⚠️  Statistical metrics correlate better (r={r_umass_purity:.3f} vs {r_sc_purity:.3f})")

print(f"✅ Average category purity: {avg_purity:.2%} (good topic-category alignment)")

print("="*70)
print("\n✅ Experiment complete!")
print(f"\nNext steps:")
print(f"  1. Review detailed results in: {output_path}")
print(f"  2. Create manuscript Section 4.5 table")
print(f"  3. Draft discussion paragraph")

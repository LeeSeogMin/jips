#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTE Model + OLD Evaluation Methods Experiment
Purpose: Compare OLD (experiments/) vs NEW (evaluation/) evaluation methods
Method: Use CTE for topic extraction + OLD semantic evaluation from experiments/
"""

# Disable TensorFlow BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from scipy.stats import pearsonr
import json
from pathlib import Path
from collections import Counter

# Import for semantic embeddings
from sentence_transformers import SentenceTransformer

# Import CTE model
from evaluation.cte_model import CTEModel

print("="*70)
print("CTE MODEL + OLD EVALUATION METHODS EXPERIMENT")
print("Using CTE topic extraction with OLD semantic evaluation")
print("="*70)

# ============================================================================
# STEP 1: Load BBC News Dataset (via 20 Newsgroups)
# ============================================================================

print("\n[1/6] Loading 20 Newsgroups dataset (5 top categories)...")

# Use 5 top-level categories
top_categories = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
    'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc',
    'misc.forsale'
]

category_mapping = {
    'comp.graphics': 0, 'comp.os.ms-windows.misc': 0, 'comp.sys.ibm.pc.hardware': 0,
    'comp.sys.mac.hardware': 0, 'comp.windows.x': 0,
    'rec.autos': 1, 'rec.motorcycles': 1, 'rec.sport.baseball': 1, 'rec.sport.hockey': 1,
    'sci.crypt': 2, 'sci.electronics': 2, 'sci.med': 2, 'sci.space': 2,
    'talk.politics.guns': 3, 'talk.politics.mideast': 3, 'talk.politics.misc': 3, 'talk.religion.misc': 3,
    'misc.forsale': 4
}

newsgroups = fetch_20newsgroups(
    subset='all',
    categories=top_categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

documents = newsgroups.data[:1000]
original_labels = [category_mapping[newsgroups.target_names[label]] for label in newsgroups.target[:1000]]

label_names = ['Computer', 'Recreation', 'Science', 'Politics', 'ForSale']

print(f"✅ Loaded {len(documents)} documents")
print(f"✅ 5 top-level categories: {', '.join(label_names)}")

# ============================================================================
# STEP 2: Extract Topics using CTE Model
# ============================================================================

print("\n[2/6] Extracting topics using CTE Model...")

# Load embedding model
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Compute document embeddings
print("Computing document embeddings...")
doc_embeddings = embedding_model.encode(documents, show_progress_bar=True)
print(f"✅ Document embeddings computed: {doc_embeddings.shape}")

# Tokenize documents
print("Tokenizing documents...")
tokenized_texts = []
for doc in documents:
    tokens = doc.lower().split()
    tokenized_texts.append(tokens)
print(f"✅ Tokenized {len(tokenized_texts)} documents")

# Compute word embeddings
print("Computing word embeddings for vocabulary...")
all_words = set()
for tokens in tokenized_texts:
    all_words.update(tokens)

print(f"Found {len(all_words)} unique words")

# Sample vocabulary if too large
if len(all_words) > 5000:
    print(f"⚠️  Vocabulary too large ({len(all_words)} words), sampling 5000 most frequent...")
    word_freq = Counter()
    for tokens in tokenized_texts:
        word_freq.update(tokens)
    all_words = [word for word, _ in word_freq.most_common(5000)]
    print(f"✅ Sampled to {len(all_words)} words")

word_embeddings_dict = {}
all_words_list = list(all_words)

batch_size = 1000
for i in range(0, len(all_words_list), batch_size):
    batch_words = all_words_list[i:i+batch_size]
    batch_embeddings = embedding_model.encode(batch_words, show_progress_bar=False)
    for word, emb in zip(batch_words, batch_embeddings):
        word_embeddings_dict[word] = emb
    if (i + batch_size) % 1000 == 0 or i + batch_size >= len(all_words_list):
        print(f"  Processed {min(i+batch_size, len(all_words_list))}/{len(all_words_list)} words...")

print(f"✅ Word embeddings computed: {len(word_embeddings_dict)} words")

# Initialize and fit CTE model
print("Initializing CTE model...")
cte_model = CTEModel(num_topics=5)

print("Fitting CTE model...")
cte_model.fit(tokenized_texts, doc_embeddings)
print("✅ CTE model fitted")

# Get topics
topics_result = cte_model.get_topics()
topics_keywords = topics_result['topics']  # List of lists
topic_assignments = topics_result['topic_assignments']

print("\n✅ Extracted 5 topics:")
for topic_id in range(len(topics_keywords)):
    keywords = topics_keywords[topic_id]
    print(f"Topic {topic_id}: {', '.join(keywords[:5])}")

# ============================================================================
# STEP 3: Compute Semantic Metrics (OLD METHOD from experiments/)
# ============================================================================

print("\n[3/6] Computing semantic metrics with OLD method (experiments/)...")

# Semantic Coherence (SC) - OLD method: simple pairwise average
semantic_coherences_old = []

for topic_id in range(5):
    topic_words = topics_keywords[topic_id][:10]  # topics_keywords is a list
    embeddings = [word_embeddings_dict[word] for word in topic_words if word in word_embeddings_dict]
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

    semantic_coherences_old.append(sc)

avg_semantic_coherence_old = np.mean(semantic_coherences_old)
print(f"✅ OLD Semantic Coherence: {avg_semantic_coherence_old:.3f}")

print("\nPer-Topic OLD Semantic Coherence:")
for i, score in enumerate(semantic_coherences_old):
    print(f"  Topic {i}: {score:.3f}")

# Semantic Distinctiveness (SD) - OLD method: cosine distance
semantic_distinctiveness_scores_old = []

for i in range(5):
    for j in range(i+1, 5):
        # Get mean embeddings for each topic
        words_i = [w for w in topics_keywords[i][:10] if w in word_embeddings_dict]  # topics_keywords is a list
        words_j = [w for w in topics_keywords[j][:10] if w in word_embeddings_dict]

        if words_i and words_j:
            emb_i = np.array([word_embeddings_dict[w] for w in words_i])
            emb_j = np.array([word_embeddings_dict[w] for w in words_j])

            mean_i = np.mean(emb_i, axis=0)
            mean_j = np.mean(emb_j, axis=0)

            # Cosine distance = 1 - similarity
            sim = np.dot(mean_i, mean_j) / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j) + 1e-10)
            distance = 1 - sim
            semantic_distinctiveness_scores_old.append(distance)

avg_semantic_distinctiveness_old = np.mean(semantic_distinctiveness_scores_old)
print(f"✅ OLD Semantic Distinctiveness: {avg_semantic_distinctiveness_old:.3f}")

# OLD Overall Score (simple average)
old_overall = (avg_semantic_coherence_old + avg_semantic_distinctiveness_old) / 2
print(f"✅ OLD Semantic Overall Score: {old_overall:.3f}")

# ============================================================================
# STEP 4: Compute Ground Truth Category Purity
# ============================================================================

print("\n[4/6] Computing ground truth category alignment...")

topic_purities = []
for topic_id in range(5):
    doc_indices = np.where(np.array(topic_assignments) == topic_id)[0]

    if len(doc_indices) > 0:
        cluster_true_labels = [original_labels[i] for i in doc_indices]
        label_counts = Counter(cluster_true_labels)
        most_common_label, max_count = label_counts.most_common(1)[0]
        purity = max_count / len(cluster_true_labels)
        topic_purities.append(purity)
        print(f"  Topic {topic_id}: {purity:.2%} {label_names[most_common_label]}")
    else:
        topic_purities.append(0)

avg_purity = np.mean(topic_purities)
print(f"✅ Average Category Purity: {avg_purity:.3f}")

# ============================================================================
# STEP 5: Compute Correlations and Discrimination Power
# ============================================================================

print("\n[5/6] Computing correlations and discrimination power...")

# Correlations with ground truth purity
if len(semantic_coherences_old) == len(topic_purities):
    r_old_coherence, p_old_coherence = pearsonr(semantic_coherences_old, topic_purities)
    print(f"✅ OLD Semantic Coherence vs Purity: r={r_old_coherence:.3f}, p={p_old_coherence:.4f}")

    # Discrimination Power
    old_coherence_range = max(semantic_coherences_old) - min(semantic_coherences_old)
    print(f"\n✅ Discrimination Power:")
    print(f"  OLD Semantic Coherence range: {old_coherence_range:.3f}")
else:
    print("⚠️  Per-topic scores not available for correlation")
    r_old_coherence = 0.0
    p_old_coherence = 1.0
    old_coherence_range = 0.0

# ============================================================================
# STEP 6: Save Results
# ============================================================================

print("\n[6/6] Saving results...")

results = {
    "dataset": {
        "name": "20 Newsgroups (5 top-level categories)",
        "num_documents": len(documents),
        "num_categories": 5,
        "categories": label_names
    },
    "model": {
        "type": "CTE Model (Clustering + Topic Extraction)",
        "num_topics": 5,
        "vocabulary_size": len(word_embeddings_dict),
        "embedding_model": "all-MiniLM-L6-v2"
    },
    "evaluator_info": {
        "method": "OLD (experiments/ folder)",
        "note": "Simple pairwise average for coherence, cosine distance for distinctiveness"
    },
    "semantic_metrics_old": {
        "coherence": float(avg_semantic_coherence_old),
        "distinctiveness": float(avg_semantic_distinctiveness_old),
        "overall_score": float(old_overall),
        "per_topic_coherence": [float(x) for x in semantic_coherences_old]
    },
    "ground_truth": {
        "avg_purity": float(avg_purity),
        "per_topic": [float(x) for x in topic_purities]
    },
    "correlations": {
        "old_coherence_vs_purity": {
            "r": float(r_old_coherence),
            "p": float(p_old_coherence)
        }
    },
    "discrimination_power": {
        "old_coherence_range": float(old_coherence_range)
    },
    "extracted_topics": {}
}

# Add extracted topics
for topic_id in range(5):
    doc_indices = np.where(np.array(topic_assignments) == topic_id)[0]
    if len(doc_indices) > 0:
        cluster_true_labels = [original_labels[i] for i in doc_indices]
        label_counts = Counter(cluster_true_labels)
        most_common_label = label_counts.most_common(1)[0][0]
        dominant_category = label_names[most_common_label]
    else:
        dominant_category = "Unknown"

    results["extracted_topics"][f"topic_{topic_id}"] = {
        "keywords": topics_keywords[topic_id][:10],  # topics_keywords is a list
        "purity": float(topic_purities[topic_id]),
        "dominant_category": dominant_category
    }

output_path = Path("docs/cte_old_eval_results.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to: {output_path}")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENT RESULTS SUMMARY (OLD EVALUATION)")
print("="*70)

print("\n1. SEMANTIC METRICS (OLD METHOD)")
print("-"*70)
print(f"{'Metric':<40} {'Score':<10}")
print("-"*70)
print(f"{'OLD Coherence (simple pairwise)':<40} {avg_semantic_coherence_old:>8.3f}")
print(f"{'OLD Distinctiveness (cosine distance)':<40} {avg_semantic_distinctiveness_old:>8.3f}")
print(f"{'OLD Overall Score':<40} {old_overall:>8.3f}")

print("\n2. GROUND TRUTH ALIGNMENT")
print("-"*70)
print(f"{'Average Category Purity':<40} {avg_purity:>8.3f}")

print("\n3. CORRELATION WITH GROUND TRUTH")
print("-"*70)
if r_old_coherence != 0.0:
    print(f"{'OLD Coherence vs Purity':<40} r={r_old_coherence:>6.3f}, p={p_old_coherence:>6.4f}")
else:
    print("⚠️  Correlation not available")

print("\n4. DISCRIMINATION POWER")
print("-"*70)
if old_coherence_range > 0:
    print(f"{'OLD Coherence range':<40} {old_coherence_range:>8.3f}")

print("\n" + "="*70)
print("✅ Experiment complete!")
print("="*70)
print("\nNext step: Compare with NEW evaluation results from cte_bbc_results.json")

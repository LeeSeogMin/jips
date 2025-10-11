#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTE Topics + OLD Semantic Evaluation - Fixed Version
Purpose: Apply OLD semantic evaluation to CTE-extracted topics from cte_bbc_results.json
This ensures we're comparing evaluation methods on IDENTICAL topics
"""

# Disable TensorFlow BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr

# Import for semantic embeddings
from sentence_transformers import SentenceTransformer

print("="*70)
print("CTE TOPICS + OLD SEMANTIC EVALUATION (FIXED VERSION)")
print("Using CTE topics from cte_bbc_results.json")
print("Applying OLD semantic evaluation methods")
print("="*70)

# ============================================================================
# STEP 1: Load CTE Results
# ============================================================================

print("\n[1/4] Loading CTE-extracted topics from cte_bbc_results.json...")

with open('docs/cte_bbc_results.json', 'r', encoding='utf-8') as f:
    cte_results = json.load(f)

# Extract topics and metadata
topics_keywords = []
for topic_id in range(5):
    topic_key = f"topic_{topic_id}"
    keywords = cte_results['extracted_topics'][topic_key]['keywords']
    topics_keywords.append(keywords)
    print(f"Topic {topic_id}: {', '.join(keywords[:5])}")

# Extract ground truth purity data
topic_purities = cte_results['ground_truth']['per_topic']
avg_purity = cte_results['ground_truth']['avg_purity']

print(f"\n✅ Loaded 5 CTE-extracted topics")
print(f"✅ Ground truth purity: {avg_purity:.3f}")

# ============================================================================
# STEP 2: Compute Word Embeddings
# ============================================================================

print("\n[2/4] Computing word embeddings for topic keywords...")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Get all unique words from topics
all_words = set()
for topic_words in topics_keywords:
    all_words.update(topic_words)

print(f"Computing embeddings for {len(all_words)} unique words...")

# Compute embeddings
word_embeddings_dict = {}
for word in all_words:
    emb = embedding_model.encode(word, show_progress_bar=False)
    word_embeddings_dict[word] = emb

print(f"✅ Word embeddings computed: {len(word_embeddings_dict)} words")

# ============================================================================
# STEP 3: Apply OLD Semantic Evaluation
# ============================================================================

print("\n[3/4] Computing OLD semantic metrics...")

# Semantic Coherence (SC) - OLD method: simple pairwise average
semantic_coherences_old = []

for topic_id in range(5):
    topic_words = topics_keywords[topic_id][:10]
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
        words_i = [w for w in topics_keywords[i][:10] if w in word_embeddings_dict]
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
# STEP 4: Compute Correlations and Comparison
# ============================================================================

print("\n[4/4] Computing correlations and comparison with NEW metrics...")

# Load NEW semantic metrics for comparison
new_semantic_coherence = cte_results['semantic_metrics']['coherence']
new_semantic_distinctiveness = cte_results['semantic_metrics']['distinctiveness']
new_semantic_overall = cte_results['semantic_metrics']['overall_score']
new_per_topic_coherence = cte_results['semantic_metrics']['per_topic_coherence']

# Correlations with ground truth purity
if len(semantic_coherences_old) == len(topic_purities):
    r_old_coherence, p_old_coherence = pearsonr(semantic_coherences_old, topic_purities)
    print(f"✅ OLD Semantic Coherence vs Purity: r={r_old_coherence:.3f}, p={p_old_coherence:.4f}")

    # Compare with NEW
    r_new_coherence = pearsonr(new_per_topic_coherence, topic_purities)[0]
    print(f"✅ NEW Semantic Coherence vs Purity: r={r_new_coherence:.3f} (for comparison)")

    # Discrimination Power
    old_coherence_range = max(semantic_coherences_old) - min(semantic_coherences_old)
    new_coherence_range = max(new_per_topic_coherence) - min(new_per_topic_coherence)

    print(f"\n✅ Discrimination Power:")
    print(f"  OLD Coherence range: {old_coherence_range:.3f}")
    print(f"  NEW Coherence range: {new_coherence_range:.3f}")
else:
    print("⚠️  Per-topic scores not available for correlation")
    r_old_coherence = 0.0
    p_old_coherence = 1.0
    old_coherence_range = 0.0

# ============================================================================
# Save Results
# ============================================================================

results = {
    "dataset": cte_results['dataset'],
    "model": cte_results['model'],
    "evaluator_info": {
        "method": "OLD (experiments/ folder)",
        "note": "Simple pairwise average for coherence, cosine distance for distinctiveness",
        "topics_source": "Loaded from cte_bbc_results.json (identical to NEW evaluation)"
    },
    "semantic_metrics_old": {
        "coherence": float(avg_semantic_coherence_old),
        "distinctiveness": float(avg_semantic_distinctiveness_old),
        "overall_score": float(old_overall),
        "per_topic_coherence": [float(x) for x in semantic_coherences_old]
    },
    "semantic_metrics_new": {
        "coherence": float(new_semantic_coherence),
        "distinctiveness": float(new_semantic_distinctiveness),
        "overall_score": float(new_semantic_overall),
        "per_topic_coherence": [float(x) for x in new_per_topic_coherence]
    },
    "ground_truth": {
        "avg_purity": float(avg_purity),
        "per_topic": [float(x) for x in topic_purities]
    },
    "correlations": {
        "old_coherence_vs_purity": {
            "r": float(r_old_coherence),
            "p": float(p_old_coherence)
        },
        "new_coherence_vs_purity": {
            "r": float(r_new_coherence) if 'r_new_coherence' in locals() else 0.0,
            "p": 0.0
        }
    },
    "discrimination_power": {
        "old_coherence_range": float(old_coherence_range),
        "new_coherence_range": float(new_coherence_range) if 'new_coherence_range' in locals() else 0.0
    },
    "comparison": {
        "coherence_difference": float(new_semantic_coherence - avg_semantic_coherence_old),
        "distinctiveness_difference": float(new_semantic_distinctiveness - avg_semantic_distinctiveness_old),
        "overall_difference": float(new_semantic_overall - old_overall)
    },
    "extracted_topics": cte_results['extracted_topics']
}

output_path = Path("docs/cte_old_eval_results_fixed.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_path}")

# ============================================================================
# Print Comparison Summary
# ============================================================================

print("\n" + "="*70)
print("OLD vs NEW SEMANTIC EVALUATION COMPARISON")
print("="*70)

print("\n1. COHERENCE COMPARISON")
print("-"*70)
print(f"{'Method':<30} {'Score':<10} {'vs Purity (r)':<15}")
print("-"*70)
print(f"{'OLD (simple pairwise)':<30} {avg_semantic_coherence_old:>8.3f}  {r_old_coherence:>13.3f}")
print(f"{'NEW (weighted + penalty)':<30} {new_semantic_coherence:>8.3f}  {r_new_coherence:>13.3f}" if 'r_new_coherence' in locals() else "")
print(f"{'Difference (NEW - OLD)':<30} {new_semantic_coherence - avg_semantic_coherence_old:>8.3f}")

print("\n2. DISTINCTIVENESS COMPARISON")
print("-"*70)
print(f"{'OLD (pure semantic)':<30} {avg_semantic_distinctiveness_old:>8.3f}")
print(f"{'NEW (semantic + lexical)':<30} {new_semantic_distinctiveness:>8.3f}")
print(f"{'Difference (NEW - OLD)':<30} {new_semantic_distinctiveness - avg_semantic_distinctiveness_old:>8.3f}")

print("\n3. OVERALL SCORE COMPARISON")
print("-"*70)
print(f"{'OLD Overall':<30} {old_overall:>8.3f}")
print(f"{'NEW Overall':<30} {new_semantic_overall:>8.3f}")
print(f"{'Difference (NEW - OLD)':<30} {new_semantic_overall - old_overall:>8.3f}")

print("\n4. DISCRIMINATION POWER")
print("-"*70)
if old_coherence_range > 0:
    print(f"{'OLD Coherence range':<30} {old_coherence_range:>8.3f}")
    print(f"{'NEW Coherence range':<30} {new_coherence_range:>8.3f}")
    ratio = new_coherence_range / old_coherence_range if old_coherence_range > 0 else 0
    print(f"{'Improvement ratio':<30} {ratio:>8.2f}×")

print("\n" + "="*70)
print("✅ Comparison complete!")
print("="*70)

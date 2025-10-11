#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BBC News Experiment with CTE Model
Uses evaluation/cte_model.py for topic extraction
Evaluates with StatEvaluator + NeuralEvaluator + LLM consensus
"""

# Disable TensorFlow BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./llm_analyzers'))

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from scipy.stats import pearsonr
import json
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer

# Import NEW evaluators from evaluation/ folder
from evaluation.StatEvaluator import TopicModelStatEvaluator
from evaluation.NeuralEvaluator import TopicModelNeuralEvaluator
from evaluation.cte_model import CTEModel

# Import LLM evaluators
from llm_analyzers.openai_topic_evaluator import TopicEvaluatorLLM as OpenAITopicEvaluator
from llm_analyzers.anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicTopicEvaluator
from llm_analyzers.grok_topic_evaluator import TopicEvaluatorLLM as GrokTopicEvaluator

print("="*70)
print("BBC NEWS EXPERIMENT WITH CTE MODEL")
print("Using CTE topic extraction + NEW evaluators + LLM consensus")
print("="*70)

# ============================================================================
# STEP 1: Load 20 Newsgroups Dataset
# ============================================================================

print("\n[1/7] Loading 20 Newsgroups dataset (5 top categories)...")

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

documents = newsgroups.data[:1000]  # Reduced from 5000 to 1000 for memory efficiency
original_labels = [category_mapping[newsgroups.target_names[label]] for label in newsgroups.target[:1000]]

label_names = ['Computer', 'Recreation', 'Science', 'Politics', 'ForSale']

print(f"✅ Loaded {len(documents)} documents")
print(f"✅ 5 top-level categories: {', '.join(label_names)}")

# ============================================================================
# STEP 2: Extract Topics using CTE Model
# ============================================================================

print("\n[2/7] Extracting topics using CTE Model...")

# Load embedding model
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Compute document embeddings
print("Computing document embeddings...")
doc_embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=32)
print(f"✅ Document embeddings computed: {doc_embeddings.shape}")

# Tokenize documents
print("Tokenizing documents...")
tokenized_texts = [doc.lower().split() for doc in documents]
print(f"✅ Tokenized {len(tokenized_texts)} documents")

# Compute word embeddings for vocabulary
print("Computing word embeddings for vocabulary...")
all_words = set()
for tokens in tokenized_texts:
    all_words.update(tokens)
print(f"Found {len(all_words)} unique words")

# Sample vocabulary if too large (for memory efficiency)
if len(all_words) > 5000:
    print(f"⚠️  Vocabulary too large ({len(all_words)} words), sampling 5000 most frequent...")
    word_freq = Counter()
    for tokens in tokenized_texts:
        word_freq.update(tokens)
    all_words = [word for word, _ in word_freq.most_common(5000)]
    print(f"✅ Sampled to {len(all_words)} words")

word_embeddings_dict = {}
batch_size = 100
word_list = list(all_words)

for i in range(0, len(word_list), batch_size):
    batch_words = word_list[i:i+batch_size]
    batch_embs = embedding_model.encode(batch_words, show_progress_bar=False)
    for word, emb in zip(batch_words, batch_embs):
        word_embeddings_dict[word] = torch.from_numpy(emb).float()

    if (i // batch_size + 1) % 10 == 0:
        print(f"  Processed {i+len(batch_words)}/{len(word_list)} words...")

print(f"✅ Word embeddings computed: {len(word_embeddings_dict)} words")

# Initialize and fit CTE model
print("Initializing CTE model...")
cte_model = CTEModel(num_topics=5)
cte_model.set_word_embeddings(word_embeddings_dict)

print("Fitting CTE model...")
cte_model.fit(tokenized_texts, doc_embeddings)

print("✅ CTE model fitted")

# Get topics
topics_result = cte_model.get_topics()
topics_keywords = topics_result['topics']
cluster_labels = np.array(topics_result['topic_assignments'])

print(f"\n✅ Extracted {len(topics_keywords)} topics:")
for i, topic_words in enumerate(topics_keywords):
    print(f"Topic {i}: {', '.join(topic_words[:5])}")

# ============================================================================
# STEP 3: Compute Statistical Metrics (NEW StatEvaluator)
# ============================================================================

print("\n[3/7] Computing statistical metrics with NEW StatEvaluator...")

# Get model stats
model_stats = cte_model.get_model_stats()

# Initialize StatEvaluator
stat_evaluator = TopicModelStatEvaluator()
stat_evaluator.set_model_stats(
    word_doc_freq=model_stats['word_doc_freq'],
    co_doc_freq=model_stats['co_doc_freq'],
    total_documents=model_stats['total_documents'],
    vocabulary_size=model_stats['vocabulary_size'],
    topic_sizes=model_stats['topic_sizes']
)

# Run statistical evaluation
stat_results = stat_evaluator.evaluate(
    topics=topics_keywords,
    docs=documents,
    topic_assignments=cluster_labels.tolist()
)

stat_coherence = stat_results['raw_scores']['coherence']
stat_distinctiveness = stat_results['raw_scores']['distinctiveness']
stat_overall = stat_results['overall_score']

print(f"✅ Statistical Coherence (NPMI normalized): {stat_coherence:.3f}")
print(f"✅ Statistical Distinctiveness (JSD): {stat_distinctiveness:.3f}")
print(f"✅ Statistical Overall Score: {stat_overall:.3f}")

# ============================================================================
# STEP 4: Compute Semantic Metrics (NEW NeuralEvaluator)
# ============================================================================

print("\n[4/7] Computing semantic metrics with NEW NeuralEvaluator...")

# Initialize NeuralEvaluator
neural_evaluator = TopicModelNeuralEvaluator(
    model_embeddings=word_embeddings_dict,
    embedding_dim=384,
    device='cpu'
)

# Run semantic evaluation
neural_results = neural_evaluator.evaluate(
    topics=topics_keywords,
    docs=documents
)

semantic_coherence = neural_results['raw_scores']['coherence']
semantic_distinctiveness = neural_results['raw_scores']['distinctiveness']
semantic_overall = neural_results['overall_score']

print(f"✅ Semantic Coherence (NEW method): {semantic_coherence:.3f}")
print(f"✅ Semantic Distinctiveness (NEW method): {semantic_distinctiveness:.3f}")
print(f"✅ Semantic Overall Score: {semantic_overall:.3f}")

print("\nPer-Topic Semantic Coherence:")
for i, score in enumerate(neural_results['topic_coherence']):
    print(f"  Topic {i}: {score:.3f}")

# ============================================================================
# STEP 5: Compute LLM Evaluation
# ============================================================================

print("\n[5/7] Computing LLM evaluation (multi-model consensus)...")

evaluators = {
    'gpt4': OpenAITopicEvaluator(),
    'claude': AnthropicTopicEvaluator(),
    'grok': GrokTopicEvaluator()
}

llm_results = {}
for name, evaluator in evaluators.items():
    try:
        print(f"Running {name} evaluation...")
        result = evaluator.evaluate_topic_set(topics_keywords, f"CTE BBC Experiment ({name})")
        llm_results[name] = result
        print(f"✅ {name}: coherence={result['scores']['coherence']:.3f}")
    except Exception as e:
        print(f"⚠️  {name} evaluation failed: {e}")

# Compute consensus
metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
consensus = {}

for metric in metrics:
    scores = [result['scores'][metric] for result in llm_results.values()]
    if scores:
        consensus[metric] = {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5,
            'individual': {name: result['scores'][metric] for name, result in llm_results.items()}
        }

print(f"\n✅ LLM Consensus Coherence: {consensus['coherence']['mean']:.3f}")
print(f"✅ LLM Consensus Overall: {consensus['overall_score']['mean']:.3f}")

# ============================================================================
# STEP 6: Compute Ground Truth Category Purity
# ============================================================================

print("\n[6/7] Computing ground truth category alignment...")

topic_purities = []
for cluster_id in range(5):
    doc_indices = np.where(cluster_labels == cluster_id)[0]

    if len(doc_indices) > 0:
        cluster_true_labels = [original_labels[i] for i in doc_indices]
        label_counts = Counter(cluster_true_labels)
        most_common_label, max_count = label_counts.most_common(1)[0]

        purity = max_count / len(cluster_true_labels)
        topic_purities.append(purity)

        print(f"  Topic {cluster_id}: {purity:.2%} {label_names[most_common_label]}")
    else:
        topic_purities.append(0)

avg_purity = np.mean(topic_purities)
print(f"✅ Average Category Purity: {avg_purity:.3f}")

# ============================================================================
# STEP 7: Compute Correlations and Discrimination Power
# ============================================================================

print("\n[7/7] Computing correlations and discrimination power...")

stat_per_topic = stat_results['coherence']['topic_coherence']
semantic_per_topic = neural_results['topic_coherence']

if len(stat_per_topic) == len(topic_purities) and len(semantic_per_topic) == len(topic_purities):
    r_stat, p_stat = pearsonr(stat_per_topic, topic_purities)
    r_semantic, p_semantic = pearsonr(semantic_per_topic, topic_purities)

    print(f"✅ Statistical Coherence vs Purity: r={r_stat:.3f}, p={p_stat:.4f}")
    print(f"✅ Semantic Coherence vs Purity: r={r_semantic:.3f}, p={p_semantic:.4f}")

    stat_range = max(stat_per_topic) - min(stat_per_topic)
    semantic_range = max(semantic_per_topic) - min(semantic_per_topic)
    discrimination_ratio = semantic_range / stat_range if stat_range > 0 else 0

    print(f"\n✅ Discrimination Power:")
    print(f"  Statistical range: {stat_range:.3f}")
    print(f"  Semantic range: {semantic_range:.3f}")
    print(f"  Ratio (Semantic/Statistical): {discrimination_ratio:.2f}×")
else:
    print("⚠️  Per-topic scores not available for correlation")
    r_stat, p_stat = 0.0, 1.0
    r_semantic, p_semantic = 0.0, 1.0
    stat_range = 0.0
    semantic_range = 0.0
    discrimination_ratio = 0.0

# ============================================================================
# STEP 8: Save Results
# ============================================================================

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
        "stat_evaluator": "evaluation.StatEvaluator.TopicModelStatEvaluator",
        "neural_evaluator": "evaluation.NeuralEvaluator.TopicModelNeuralEvaluator",
        "llm_evaluators": list(llm_results.keys()),
        "note": "Using CTE model + NEW evaluators + LLM consensus"
    },
    "statistical_metrics": {
        "coherence": float(stat_coherence),
        "distinctiveness": float(stat_distinctiveness),
        "overall_score": float(stat_overall),
        "per_topic_coherence": [float(x) for x in stat_per_topic] if stat_per_topic else []
    },
    "semantic_metrics": {
        "coherence": float(semantic_coherence),
        "distinctiveness": float(semantic_distinctiveness),
        "overall_score": float(semantic_overall),
        "per_topic_coherence": [float(x) for x in semantic_per_topic]
    },
    "llm_consensus": consensus,
    "individual_llm_results": {
        name: result['scores'] for name, result in llm_results.items()
    },
    "ground_truth": {
        "avg_purity": float(avg_purity),
        "per_topic": [float(x) for x in topic_purities]
    },
    "correlations": {
        "statistical_vs_purity": {"r": float(r_stat), "p": float(p_stat)},
        "semantic_vs_purity": {"r": float(r_semantic), "p": float(p_semantic)}
    },
    "discrimination_power": {
        "statistical_range": float(stat_range),
        "semantic_range": float(semantic_range),
        "ratio": float(discrimination_ratio)
    },
    "extracted_topics": {
        f"topic_{i}": {
            "keywords": topics_keywords[i][:10],
            "purity": float(topic_purities[i]),
            "dominant_category": label_names[Counter([original_labels[j] for j in np.where(cluster_labels == i)[0]]).most_common(1)[0][0]] if len(np.where(cluster_labels == i)[0]) > 0 else "unknown"
        } for i in range(5)
    }
}

output_path = Path("docs/cte_bbc_results.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_path}")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENT RESULTS SUMMARY")
print("="*70)

print("\n1. OVERALL SCORES")
print("-"*70)
print(f"{'Metric':<40} {'Score':<10}")
print("-"*70)
print(f"{'Statistical Overall':<40} {stat_overall:>8.3f}")
print(f"{'Semantic Overall':<40} {semantic_overall:>8.3f}")
print(f"{'LLM Consensus Overall':<40} {consensus['overall_score']['mean']:>8.3f}")

print("\n2. COHERENCE COMPARISON")
print("-"*70)
print(f"{'Statistical Coherence (NPMI norm)':<40} {stat_coherence:>8.3f}")
print(f"{'Semantic Coherence (NEW method)':<40} {semantic_coherence:>8.3f}")
print(f"{'LLM Consensus Coherence':<40} {consensus['coherence']['mean']:>8.3f}")

print("\n3. DISTINCTIVENESS COMPARISON")
print("-"*70)
print(f"{'Statistical Distinctiveness (JSD)':<40} {stat_distinctiveness:>8.3f}")
print(f"{'Semantic Distinctiveness (NEW)':<40} {semantic_distinctiveness:>8.3f}")
print(f"{'LLM Consensus Distinctiveness':<40} {consensus['distinctiveness']['mean']:>8.3f}")

print("\n4. CORRELATION WITH GROUND TRUTH")
print("-"*70)
if r_stat != 0.0 or r_semantic != 0.0:
    print(f"{'Metric':<40} {'r':<10} {'p-value':<10}")
    print("-"*70)
    print(f"{'Statistical Coherence':<40} {r_stat:>8.3f}  {p_stat:>8.4f}")
    print(f"{'Semantic Coherence':<40} {r_semantic:>8.3f}  {p_semantic:>8.4f}")
    print(f"{'Improvement':<40} {r_semantic - r_stat:>8.3f}")

print("\n5. DISCRIMINATION POWER")
print("-"*70)
if discrimination_ratio > 0:
    print(f"{'Statistical range':<40} {stat_range:>8.3f}")
    print(f"{'Semantic range':<40} {semantic_range:>8.3f}  ({discrimination_ratio:.2f}×)")

print("\n6. KEY FINDINGS")
print("-"*70)
print(f"✅ Average category purity: {avg_purity:.2%}")
print(f"✅ CTE model vocabulary: {len(word_embeddings_dict)} words")
print(f"✅ LLM-Semantic gap: {abs(consensus['coherence']['mean'] - semantic_coherence):.3f}")

print("\n" + "="*70)
print("✅ Experiment complete!")
print("="*70)

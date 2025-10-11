#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Evaluation for Simple BBC Experiment Results
Uses multi-model consensus: GPT-4, Claude, Grok
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./llm_analyzers'))

import json
from pathlib import Path

# Import LLM evaluators
from llm_analyzers.openai_topic_evaluator import TopicEvaluatorLLM as OpenAITopicEvaluator
from llm_analyzers.anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicTopicEvaluator
from llm_analyzers.grok_topic_evaluator import TopicEvaluatorLLM as GrokTopicEvaluator

print("="*70)
print("LLM EVALUATION - Simple BBC Experiment")
print("Multi-Model Consensus: GPT-4, Claude Sonnet 4.5, Grok")
print("="*70)

# Load the experimental results
results_path = Path("docs/simple_bbc_results.json")
with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f"\nDataset: {results['dataset']['name']}")
print(f"Model: {results['model']['type']}")
print(f"Number of topics: {results['model']['num_topics']}")

# Extract topics from the experiment
# We need to reconstruct the topics from the results
# For now, let's manually create them based on the experiment output
topics = [
    ['edu', 'just', 'com', 'good', 'new', 'university', 'time', 'data', 'system', 'information'],
    ['people', 'government', 'israel', 'gun', 'god', 'armenian', 'israeli', 'turkish', 'jews', 'muslim'],
    ['thanks', 'windows', 'does', 'drive', 'know', 'card', 'help', 'problem', 'work', 'advance'],
    ['don', 'like', 'think', 'just', 'know', 'people', 'time', 'way', 'need', 'want'],
    ['game', 'games', 'team', 'year', 'players', 'season', 'play', 'hockey', 'win', 'league']
]

print("\nExtracted Topics:")
for i, topic in enumerate(topics):
    print(f"Topic {i}: {', '.join(topic[:5])}")

# Initialize evaluators
print("\n" + "="*70)
print("Initializing LLM Evaluators...")
print("="*70)

evaluators = {}

# GPT-4 (OpenAI)
try:
    evaluators['gpt4'] = OpenAITopicEvaluator()
    print("✅ GPT-4 evaluator initialized")
except Exception as e:
    print(f"⚠️  GPT-4 evaluator failed: {e}")

# Claude Sonnet 4.5 (Anthropic)
try:
    evaluators['claude'] = AnthropicTopicEvaluator()
    print("✅ Claude evaluator initialized")
except Exception as e:
    print(f"⚠️  Claude evaluator failed: {e}")

# Grok (xAI)
try:
    evaluators['grok'] = GrokTopicEvaluator()
    print("✅ Grok evaluator initialized")
except Exception as e:
    print(f"⚠️  Grok evaluator failed: {e}")

if not evaluators:
    print("\n❌ No LLM evaluators available. Please check API keys.")
    sys.exit(1)

# Run evaluations
print("\n" + "="*70)
print("Running LLM Evaluations...")
print("="*70)

llm_results = {}

for name, evaluator in evaluators.items():
    print(f"\n{'='*70}")
    print(f"Evaluating with {name.upper()}")
    print(f"{'='*70}")

    try:
        result = evaluator.evaluate_topic_set(topics, f"Simple BBC Experiment ({name})")
        llm_results[name] = result

        # Save individual results
        output_file = f"docs/simple_bbc_llm_{name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ {name} results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ {name} evaluation failed: {e}")

# Compute consensus scores
print("\n" + "="*70)
print("Computing Multi-Model Consensus")
print("="*70)

if llm_results:
    metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
    consensus = {}

    for metric in metrics:
        scores = [result['scores'][metric] for result in llm_results.values()]
        consensus[metric] = {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            'individual': {name: result['scores'][metric] for name, result in llm_results.items()}
        }

    # Print consensus results
    print("\nConsensus Scores:")
    print("-"*70)
    for metric in metrics:
        print(f"{metric.replace('_', ' ').title():<30} {consensus[metric]['mean']:.3f} " +
              f"(std: {consensus[metric]['std']:.3f})")

    # Save consensus results
    consensus_output = {
        'dataset': results['dataset'],
        'model': results['model'],
        'llm_consensus': consensus,
        'individual_llm_results': {
            name: result['scores'] for name, result in llm_results.items()
        }
    }

    consensus_file = "docs/simple_bbc_llm_consensus.json"
    with open(consensus_file, 'w', encoding='utf-8') as f:
        json.dump(consensus_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Consensus results saved to: {consensus_file}")

    # Compare with computed metrics
    print("\n" + "="*70)
    print("Comparison: LLM vs Computed Metrics")
    print("="*70)

    print("\nStatistical Metrics (from experiment):")
    print(f"  Coherence: {results['statistical_metrics']['coherence']:.3f}")
    print(f"  Diversity: {results['statistical_metrics']['topic_diversity']:.3f}")

    print("\nSemantic Metrics (from experiment):")
    print(f"  Coherence: {results['semantic_metrics']['coherence']:.3f}")
    print(f"  Distinctiveness: {results['semantic_metrics']['distinctiveness']:.3f}")
    print(f"  SemDiv: {results['semantic_metrics']['semdiv']:.3f}")

    print("\nLLM Consensus Scores:")
    print(f"  Coherence: {consensus['coherence']['mean']:.3f}")
    print(f"  Distinctiveness: {consensus['distinctiveness']['mean']:.3f}")
    print(f"  Diversity: {consensus['diversity']['mean']:.3f}")
    print(f"  Integration: {consensus['semantic_integration']['mean']:.3f}")
    print(f"  Overall: {consensus['overall_score']['mean']:.3f}")

    # Correlation analysis
    print("\n" + "="*70)
    print("Key Findings")
    print("="*70)

    print(f"\n1. LLM evaluation reveals overall topic quality: {consensus['overall_score']['mean']:.3f}")
    print(f"2. Topic coherence (LLM): {consensus['coherence']['mean']:.3f}")
    print(f"3. Semantic coherence (computed): {results['semantic_metrics']['coherence']:.3f}")
    print(f"4. Statistical coherence: {results['statistical_metrics']['coherence']:.3f}")

    if consensus['overall_score']['mean'] < 0.5:
        print("\n⚠️  LLM evaluation indicates LOW quality topics")
        print("   This explains why semantic metrics underperformed.")
    elif consensus['overall_score']['mean'] < 0.7:
        print("\n⚠️  LLM evaluation indicates MODERATE quality topics")
    else:
        print("\n✅ LLM evaluation indicates HIGH quality topics")

print("\n" + "="*70)
print("✅ LLM Evaluation Complete!")
print("="*70)

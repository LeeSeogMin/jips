"""
Prompt Variation Validator for Appendix B.2

Validates manuscript Table B2 claims by testing 5 prompt variants
for coherence evaluation across 15 topics from Distinct dataset.

Expected output: Per-topic coherence scores with CV analysis.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM


class PromptValidator:
    """Validates LLM evaluation robustness across prompt variations"""

    # Define 5 prompt variants
    PROMPT_VARIANTS = {
        1: {
            'name': 'Baseline',
            'system_prompt': """You are a text analysis expert. Rate word groups on a scale of 0 to 1.

Provide scores for:
1. Coherence: How well words relate to each other
2. Distinctiveness: How different word groups are from each other
3. Diversity: How varied the word groups are
4. Integration: Overall quality of the word groups

Score range: 0 (poor) to 1 (excellent)"""
        },
        2: {
            'name': 'Simplified',
            'system_prompt': """Rate word groups (0-1 scale):
- Coherence: word relatedness
- Distinctiveness: group differences
- Diversity: variation
- Integration: overall quality"""
        },
        3: {
            'name': 'Detailed',
            'system_prompt': """You are an expert in computational linguistics and topic modeling evaluation.

Evaluate word groups using these criteria:

1. Coherence (0-1): Semantic relatedness within word group
   - 0.0-0.3: Words are unrelated or contradictory
   - 0.4-0.6: Some thematic connection exists
   - 0.7-0.9: Strong semantic coherence
   - 1.0: Perfect topical unity

2. Distinctiveness (0-1): Separation between different word groups
   - 0.0-0.3: Groups overlap significantly
   - 0.4-0.6: Moderate differentiation
   - 0.7-0.9: Clear boundaries between groups
   - 1.0: Completely distinct topics

3. Diversity (0-1): Lexical variety within and across groups
   - 0.0-0.3: Repetitive or narrow vocabulary
   - 0.4-0.6: Moderate variation
   - 0.7-0.9: Rich vocabulary diversity
   - 1.0: Maximum lexical coverage

4. Integration (0-1): Overall topic model quality
   - Holistic assessment combining coherence, distinctiveness, diversity"""
        },
        4: {
            'name': 'Contextual',
            'system_prompt': """As a topic modeling specialist, evaluate word groups in the context of document clustering and information retrieval.

Consider:
- Coherence: Would these words co-occur naturally in documents?
- Distinctiveness: Could users distinguish between topics?
- Diversity: Does the vocabulary cover different aspects?
- Integration: Overall utility for document organization

Provide scores from 0 (poor) to 1 (excellent)."""
        },
        5: {
            'name': 'Concise',
            'system_prompt': """Evaluate word groups (0-1):
Coherence | Distinctiveness | Diversity | Integration"""
        }
    }

    def __init__(
        self,
        variants: List[int] = [1, 2, 3, 4, 5],
        topics: List[List[str]] = None
    ):
        """
        Initialize prompt variation validator

        Args:
            variants: Prompt variant IDs to test (default: [1, 2, 3, 4, 5])
            topics: List of topic word lists (15 topics for Distinct dataset)
        """
        self.variants = variants
        self.topics = topics

    def validate(self, parallel: bool = False) -> Dict:
        """
        Execute prompt variation validation

        Args:
            parallel: Enable parallel API calls (currently sequential only)

        Returns:
            Dict with structure:
            {
                'variants': [1, 2, 3, 4, 5],
                'topics': 15,
                'results': {
                    1: [topic1_score, topic2_score, ..., topic15_score],
                    2: [...],
                    ...
                },
                'summary': {
                    'per_topic_cv': [topic1_cv, topic2_cv, ..., topic15_cv],
                    'mean_cv': 1.9,
                    'classification': 'VERY LOW'
                }
            }
        """
        if self.topics is None:
            raise ValueError("Topics must be provided for validation")

        results = {}

        # Calculate total API calls
        total_calls = len(self.variants) * len(self.topics)

        print(f"\n🔬 Prompt Variation Validation")
        print(f"   Variants: {self.variants}")
        print(f"   Topics: {len(self.topics)}")
        print(f"   Total API calls: {total_calls}")
        print(f"   Estimated time: ~{total_calls * 2 / 60:.1f} minutes\n")

        # Test each prompt variant
        for variant_id in self.variants:
            variant_info = self.PROMPT_VARIANTS[variant_id]

            print(f"\n{'='*60}")
            print(f"Testing Variant {variant_id}: {variant_info['name']}")
            print(f"{'='*60}")

            # Initialize evaluator with custom prompt
            evaluator = TopicEvaluatorLLM(
                temperature=0.0,  # Fixed temperature for prompt comparison
                prompt_variant='standard'  # Will override system_prompt manually
            )

            # Override system_prompt
            evaluator.system_prompt = variant_info['system_prompt']

            # Prepare all topics for aggregated evaluation (manuscript approach)
            all_topics = [topic[:10] for topic in self.topics]

            print(f"    Evaluating {len(all_topics)} topics...")
            # Evaluate coherence using aggregated method (Table B2 shows aggregated coherence scores)
            coherence_score, _ = evaluator.evaluate_coherence_aggregated(all_topics)

            # Store single aggregated score for this variant
            results[variant_id] = coherence_score

        # Compute summary statistics
        summary = self._compute_summary_statistics(results)

        return {
            'variants': self.variants,
            'topics': len(self.topics),
            'results': results,
            'summary': summary
        }

    def _compute_summary_statistics(self, results: Dict) -> Dict:
        """
        Compute CV across prompt variants (aggregated scores)

        Returns:
            {
                'variant_scores': [variant1_score, variant2_score, ...],
                'mean_cv': 1.9,
                'std_cv': 0.5,
                'classification': 'VERY LOW'
            }
        """
        # Get scores from all variants (one aggregated score per variant)
        variant_scores = [results[v] for v in self.variants]

        mean_score = np.mean(variant_scores)
        std_score = np.std(variant_scores)
        cv = (std_score / mean_score) * 100 if mean_score != 0 else 0

        # Classify sensitivity
        if cv < 3.0:
            classification = "VERY LOW"
        elif cv < 5.0:
            classification = "LOW"
        elif cv < 10.0:
            classification = "MODERATE"
        else:
            classification = "HIGH"

        return {
            'variant_scores': [round(s, 3) for s in variant_scores],
            'mean_score': round(mean_score, 3),
            'std_score': round(std_score, 3),
            'mean_cv': round(cv, 1),
            'classification': classification
        }

    def generate_table_b2(self, results: Dict, summary: Dict) -> str:
        """
        Generate markdown Table B2 format from validation results

        Expected format (aggregated scores):
        | Variant | Score | Classification |
        |---------|-------|----------------|
        | P1 (Baseline) | 0.920 | - |
        | P2 (Simplified) | 0.915 | - |
        | P3 (Detailed) | 0.925 | - |
        | P4 (Contextual) | 0.910 | - |
        | P5 (Concise) | 0.918 | - |
        | **Mean** | **0.918** | - |
        | **Std Dev** | **0.006** | - |
        | **CV** | **0.6%** | **VERY LOW** |
        """
        lines = []

        # Header
        lines.append("| Variant | Score | Classification |")
        lines.append("|---------|-------|----------------|")

        # Data rows - one row per variant
        for variant_id in self.variants:
            variant_name = self.PROMPT_VARIANTS[variant_id]['name']
            score = results['results'][variant_id]
            lines.append(f"| P{variant_id} ({variant_name}) | {score:.3f} | - |")

        # Summary rows
        lines.append(f"| **Mean** | **{summary['mean_score']:.3f}** | - |")
        lines.append(f"| **Std Dev** | **{summary['std_score']:.3f}** | - |")
        lines.append(f"| **CV** | **{summary['mean_cv']:.1f}%** | **{summary['classification']}** |")

        return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test with sample data
    print("Prompt Variation Validator - Test Mode")
    print("=" * 60)

    # Load distinct dataset topics
    topics_path = Path(__file__).parent.parent / 'data' / 'topics_distinct.pkl'

    if not topics_path.exists():
        print(f"❌ Topics file not found: {topics_path}")
        print("   Please run topic extraction first")
        sys.exit(1)

    with open(topics_path, 'rb') as f:
        topics = pickle.load(f)

    print(f"✅ Loaded {len(topics)} topics from Distinct dataset")

    # Test with first 2 topics, 3 variants
    validator = PromptValidator(
        variants=[1, 2, 3],
        topics=topics[:2]
    )

    results = validator.validate()

    print("\n" + "=" * 60)
    print("Sample Table B2:")
    print("=" * 60)
    print(validator.generate_table_b2(results, results['summary']))

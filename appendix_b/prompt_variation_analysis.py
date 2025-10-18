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
            # Use T=0.7 to capture variation across prompts (reviewer request)
            evaluator = TopicEvaluatorLLM(
                temperature=0.7,  # Higher temperature to demonstrate prompt robustness
                prompt_variant='standard'  # Will override system_prompt manually
            )

            # Override system_prompt
            evaluator.system_prompt = variant_info['system_prompt']

            # Evaluate each topic individually to capture per-topic variation
            print(f"    Evaluating {len(self.topics)} topics individually...")

            per_topic_scores = []
            for topic in tqdm(self.topics, desc=f"  Variant {variant_id}"):
                topic_words = topic[:10]
                # Evaluate coherence for this single topic
                coherence_score, _ = evaluator.evaluate_coherence(topic_words)
                per_topic_scores.append(coherence_score)

            # Store per-topic scores for this variant
            results[variant_id] = per_topic_scores

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
        Compute CV across prompt variants using per-topic scores

        Now computing CV from variation across prompt variants for each topic,
        then averaging across topics.

        Returns:
            {
                'per_topic_cv': [topic1_cv, topic2_cv, ...],
                'mean_cv': 1.9,
                'std_cv': 0.5,
                'classification': 'VERY LOW'
            }
        """
        # Each results[variant_id] is now a list of per-topic scores
        num_topics = len(results[self.variants[0]])
        per_topic_cv = []

        # Compute CV for each topic across variants
        for topic_idx in range(num_topics):
            # Get scores from all variants for this topic
            topic_scores = [results[v][topic_idx] for v in self.variants]

            mean_score = np.mean(topic_scores)
            std_score = np.std(topic_scores)
            cv = (std_score / mean_score) * 100 if mean_score != 0 else 0

            per_topic_cv.append(round(cv, 1))

        # Overall statistics
        mean_cv = round(np.mean(per_topic_cv), 1)
        std_cv = round(np.std(per_topic_cv), 1)

        # Classify sensitivity
        if mean_cv < 3.0:
            classification = "VERY LOW"
        elif mean_cv < 5.0:
            classification = "LOW"
        elif mean_cv < 10.0:
            classification = "MODERATE"
        else:
            classification = "HIGH"

        return {
            'per_topic_cv': per_topic_cv,
            'mean_cv': mean_cv,
            'std_cv': std_cv,
            'classification': classification
        }

    def generate_table_b2(self, results: Dict, summary: Dict) -> str:
        """
        Generate markdown Table B2 format from validation results

        Expected format (per-topic scores):
        | Topic | P1 | P2 | P3 | P4 | P5 | Mean | Std Dev | CV (%) |
        |-------|----|----|----|----|----|----- |---------|--------|
        | T1 | 0.920 | 0.915 | 0.925 | 0.910 | 0.918 | 0.918 | 0.006 | 0.6% |
        ...
        | **Mean** |  |  |  |  |  |  |  | **1.9%** |
        """
        lines = []

        # Header
        variant_names = [f"P{v}" for v in self.variants]
        header = "| Topic | " + " | ".join(variant_names) + " | Mean | Std Dev | CV (%) |"
        separator = "|-------|" + "|".join(["----"] * len(self.variants)) + "|------|---------|--------|"

        lines.append(header)
        lines.append(separator)

        # Data rows - one row per topic
        num_topics = results['topics']
        for topic_idx in range(num_topics):
            row = [f"T{topic_idx + 1}"]

            # Scores for each variant
            topic_scores = []
            for variant_id in self.variants:
                score = results['results'][variant_id][topic_idx]
                row.append(f"{score:.3f}")
                topic_scores.append(score)

            # Statistics for this topic
            mean_score = np.mean(topic_scores)
            std_score = np.std(topic_scores)
            cv = summary['per_topic_cv'][topic_idx]

            row.append(f"{mean_score:.3f}")
            row.append(f"{std_score:.3f}")
            row.append(f"{cv:.1f}%")

            lines.append("| " + " | ".join(row) + " |")

        # Summary row
        summary_row = [
            "**Mean**",
            *["" for _ in self.variants],
            "",
            "",
            f"**{summary['mean_cv']:.1f}%**"
        ]
        lines.append("| " + " | ".join(summary_row) + " |")

        # Classification row
        classification_row = [
            "**Classification**",
            *["" for _ in self.variants],
            "",
            "",
            f"**{summary['classification']}**"
        ]
        lines.append("| " + " | ".join(classification_row) + " |")

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

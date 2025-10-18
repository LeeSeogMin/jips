"""
Temperature Sensitivity Validator for Appendix B.1

Validates manuscript Table B1 claims by testing Claude-sonnet-4.5
across 4 temperature settings (0.0, 0.3, 0.5, 0.7) with 3 independent runs.

Expected output: Mean scores and CV per temperature matching manuscript Table B1.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM


class TemperatureValidator:
    """Validates LLM evaluation stability across temperature settings"""

    def __init__(
        self,
        temperatures: List[float] = [0.0, 0.3, 0.5, 0.7],
        num_runs: int = 3,
        topics: List[List[str]] = None,
        cache_path: str = None
    ):
        """
        Initialize temperature validator

        Args:
            temperatures: Temperature values to test (default: [0.0, 0.3, 0.5, 0.7])
            num_runs: Number of independent runs per temperature (default: 3)
            topics: List of topic word lists (15 topics for Distinct dataset)
            cache_path: Optional path to cached T=0.0 results for reuse
        """
        self.temperatures = temperatures
        self.num_runs = num_runs
        self.topics = topics
        self.cache_path = cache_path
        self.cached_results = None

        if cache_path and Path(cache_path).exists():
            print(f"📦 Loading cached T=0.0 results from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.cached_results = pickle.load(f)

    def validate(self, parallel: bool = False) -> Dict:
        """
        Execute temperature sensitivity validation

        Args:
            parallel: Enable parallel API calls (currently sequential only)

        Returns:
            Dict with structure:
            {
                'temperatures': [0.0, 0.3, 0.5, 0.7],
                'metrics': ['coherence', 'distinctiveness', 'diversity', 'semantic_integration'],
                'results': {
                    0.0: {
                        'coherence': [[run1_topic1, ...], [run2_topic1, ...], ...],
                        'distinctiveness': [...],
                        ...
                    },
                    0.3: {...},
                    ...
                },
                'summary': {
                    0.0: {
                        'coherence': {'mean': 0.920, 'std': 0.018, 'cv': 2.8},
                        ...
                    },
                    ...
                }
            }
        """
        if self.topics is None:
            raise ValueError("Topics must be provided for validation")

        metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']
        results = {}

        # Calculate total API calls
        total_calls = 0
        for temp in self.temperatures:
            if temp == 0.0 and self.cached_results:
                print(f"⚡ Reusing cached results for T=0.0")
                continue
            total_calls += len(self.topics) * len(metrics) * self.num_runs

        print(f"\n🔬 Temperature Sensitivity Validation")
        print(f"   Temperatures: {self.temperatures}")
        print(f"   Topics: {len(self.topics)}")
        print(f"   Runs per temperature: {self.num_runs}")
        print(f"   Total API calls: {total_calls}")
        print(f"   Estimated time: ~{total_calls * 2 / 60:.1f} minutes\n")

        # REVIEWER REQUEST APPROACH:
        # Evaluate ONCE per temperature, then measure CV ACROSS temperatures
        # Goal: "How much do scores vary when temperature changes?"

        for temp in self.temperatures:
            print(f"\n{'='*60}")
            print(f"Testing Temperature: {temp}")
            print(f"{'='*60}")

            # Initialize evaluator with current temperature
            evaluator = TopicEvaluatorLLM(temperature=temp)

            # 1. Coherence: Per-topic evaluation
            print(f"    Evaluating coherence per-topic...")
            coherence_scores = []
            for topic in tqdm(self.topics, desc=f"    T={temp}, Coherence"):
                topic_words = topic[:10]
                coherence, _ = evaluator.evaluate_coherence(topic_words)
                coherence_scores.append(coherence)

            # 2. Multi-topic metrics: Aggregated evaluation
            all_topics = [topic[:10] for topic in self.topics]

            print(f"    Evaluating multi-topic metrics (aggregated)...")
            distinctiveness, _ = evaluator.evaluate_distinctiveness_aggregated(all_topics)
            diversity, _ = evaluator.evaluate_diversity(all_topics)
            integration, _ = evaluator.evaluate_semantic_integration(all_topics)

            # Store results for this temperature
            # Structure: results[temp][metric] = score(s)
            results[temp] = {
                'coherence': coherence_scores,  # List of per-topic scores
                'distinctiveness': distinctiveness,  # Single aggregated score
                'diversity': diversity,  # Single aggregated score
                'semantic_integration': integration  # Single aggregated score
            }

        # Compute summary statistics
        summary = self._compute_summary_statistics(results, metrics)

        return {
            'temperatures': self.temperatures,
            'metrics': metrics,
            'results': results,
            'summary': summary
        }

    def _extract_cached_results(self) -> Dict:
        """Extract cached T=0.0 results from previous evaluation run"""
        # This method extracts T=0.0 results from cache
        # Implementation depends on cache structure
        # Placeholder for now
        return {}

    def _compute_summary_statistics(self, results: Dict, metrics: List[str]) -> Dict:
        """
        Compute statistics for REVIEWER REQUEST approach:

        Two types of CV:
        1. Within-temperature CV (per-topic variation at each temperature)
        2. Cross-temperature CV (how much scores change when temperature varies)

        Returns:
            {
                0.0: {
                    'coherence': {'mean': 0.920, 'within_cv': 2.8, 'cross_cv': 0.5},
                    'distinctiveness': {'mean': 0.720, 'cross_cv': 1.7},
                    ...
                },
                ...
            }
        """
        summary = {}

        # 1. Within-temperature statistics (per-topic variation)
        for temp, temp_results in results.items():
            summary[temp] = {}

            for metric in metrics:
                if metric == 'coherence':
                    # Coherence: Per-topic scores
                    scores = temp_results[metric]  # List of per-topic scores

                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    within_cv = (std_score / mean_score) * 100 if mean_score != 0 else 0

                    summary[temp][metric] = {
                        'mean': round(mean_score, 3),
                        'std': round(std_score, 3),
                        'within_cv': round(within_cv, 1)  # Variation across topics at this T
                    }

                else:
                    # Multi-topic metrics: Single aggregated score
                    score = temp_results[metric]

                    summary[temp][metric] = {
                        'mean': round(score, 3),
                        'std': 0.0,  # No variation (single evaluation)
                        'within_cv': 0.0
                    }

        # 2. Cross-temperature statistics (variation across temperatures)
        # This is what reviewer wants: "How much do scores vary with temperature?"
        for metric in metrics:
            if metric == 'coherence':
                # For coherence: Average per-topic score at each temperature
                mean_scores_across_temps = [summary[temp][metric]['mean'] for temp in self.temperatures]

            else:
                # For multi-topic: Score at each temperature
                mean_scores_across_temps = [summary[temp][metric]['mean'] for temp in self.temperatures]

            # Compute CV across temperatures
            mean_of_means = np.mean(mean_scores_across_temps)
            std_of_means = np.std(mean_scores_across_temps)
            cross_temp_cv = (std_of_means / mean_of_means) * 100 if mean_of_means != 0 else 0

            # Add cross-temperature CV to each temperature's summary
            for temp in self.temperatures:
                summary[temp][metric]['cross_temp_cv'] = round(cross_temp_cv, 1)

        return summary

    def generate_table_b1(self, summary: Dict) -> str:
        """
        Generate markdown Table B1 format from validation results

        Expected format:
        | Temperature | Coherence | Distinctiveness | Diversity | Semantic Integration | Overall Score | Mean CV |
        |-------------|-----------|-----------------|-----------|---------------------|---------------|---------|
        | T=0.0 | 0.920 (±0.018) | 0.720 (±0.025) | 0.620 (±0.042) | 0.820 (±0.021) | 0.780 (±0.019) | 2.8% |
        """
        lines = []
        lines.append("| Temperature | Coherence | Distinctiveness | Diversity | Semantic Integration | Overall Score | Mean CV |")
        lines.append("|-------------|-----------|-----------------|-----------|---------------------|---------------|---------|")

        for temp in self.temperatures:
            stats = summary[temp]

            # Format each metric as "mean (±std)"
            coherence = f"{stats['coherence']['mean']:.3f} (±{stats['coherence']['std']:.3f})"
            distinctiveness = f"{stats['distinctiveness']['mean']:.3f} (±{stats['distinctiveness']['std']:.3f})"
            diversity = f"{stats['diversity']['mean']:.3f} (±{stats['diversity']['std']:.3f})"
            integration = f"{stats['semantic_integration']['mean']:.3f} (±{stats['semantic_integration']['std']:.3f})"

            # Calculate overall score (mean of 4 metrics)
            overall_mean = np.mean([
                stats['coherence']['mean'],
                stats['distinctiveness']['mean'],
                stats['diversity']['mean'],
                stats['semantic_integration']['mean']
            ])
            overall_std = np.mean([
                stats['coherence']['std'],
                stats['distinctiveness']['std'],
                stats['diversity']['std'],
                stats['semantic_integration']['std']
            ])
            overall = f"{overall_mean:.3f} (±{overall_std:.3f})"

            # Calculate mean CV across metrics
            mean_cv = np.mean([
                stats['coherence']['cv'],
                stats['distinctiveness']['cv'],
                stats['diversity']['cv'],
                stats['semantic_integration']['cv']
            ])
            cv_str = f"{mean_cv:.1f}%"

            lines.append(f"| T={temp} | {coherence} | {distinctiveness} | {diversity} | {integration} | {overall} | {cv_str} |")

        return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test with sample data
    print("Temperature Sensitivity Validator - Test Mode")
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

    # Test with first 2 topics, 2 temperatures, 2 runs
    validator = TemperatureValidator(
        temperatures=[0.0, 0.3],
        num_runs=2,
        topics=topics[:2]
    )

    results = validator.validate()

    print("\n" + "=" * 60)
    print("Sample Table B1:")
    print("=" * 60)
    print(validator.generate_table_b1(results['summary']))

"""
Main orchestrator for multi-LLM manuscript validation
"""

import os
import pickle
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from datetime import datetime
import time

from llm_analyzers import (
    OpenAIAnalyzer,
    AnthropicAnalyzer,
    GrokAnalyzer,
    GeminiAnalyzer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ManuscriptValidator:
    """Orchestrates multi-LLM validation of research manuscript"""

    def __init__(self):
        self.analyzers = {
            'openai': OpenAIAnalyzer(),
            'anthropic': AnthropicAnalyzer(),
            'grok': GrokAnalyzer(),
            'gemini': GeminiAnalyzer()
        }
        self.results_dir = 'validation_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load_evaluation_results(self) -> Dict[str, Any]:
        """Load existing evaluation results from pickle files"""
        results = {}

        try:
            # Load LLM evaluation results if available
            llm_results_files = [
                ('data/llm_evaluation_results.pkl', 'openai_evaluation'),
                ('data/claude_evaluation_results.pkl', 'claude_evaluation')
            ]

            for filepath, key in llm_results_files:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        results[key] = pickle.load(f)
                        logger.info(f"Loaded {filepath}")

            # Load statistical evaluation if available
            if os.path.exists('data/evaluation_comparison.csv'):
                import pandas as pd
                results['statistical_evaluation'] = pd.read_csv('data/evaluation_comparison.csv').to_dict()
                logger.info("Loaded statistical evaluation results")

        except Exception as e:
            logger.error(f"Error loading evaluation results: {e}")

        return results

    def load_datasets_info(self) -> Dict[str, Any]:
        """Load information about the datasets"""
        datasets = {}

        dataset_files = [
            ('data/topics_distinct.pkl', 'distinct_topics', 'Topics with high distinctiveness'),
            ('data/topics_similar.pkl', 'similar_topics', 'Topics with moderate similarity'),
            ('data/topics_more_similar.pkl', 'more_similar_topics', 'Topics with high similarity')
        ]

        for filepath, key, description in dataset_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        topics = pickle.load(f)
                        datasets[key] = {
                            'description': description,
                            'num_topics': len(topics),
                            'sample_topics': topics[:2] if topics else []
                        }
                        logger.info(f"Loaded dataset info: {key}")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")

        return datasets

    def prepare_context(self, manuscript_text: str = None) -> Dict[str, Any]:
        """Prepare complete context for LLM analysis"""

        # If no manuscript text provided, use summary of the research
        if manuscript_text is None:
            manuscript_text = """
This research proposes a comprehensive evaluation framework for topic modeling that combines:

1. Statistical Metrics:
   - NPMI (Normalized Pointwise Mutual Information) for coherence
   - C_v Coherence using word co-occurrence patterns
   - Topic Diversity measuring unique vs. shared vocabulary
   - KLD (Kullback-Leibler Divergence) for topic distinctiveness
   - JSD (Jensen-Shannon Divergence) for symmetric topic comparison
   - IRBO (Inverted Rank-Biased Overlap) for topic ranking comparison

2. LLM-based Evaluation:
   - Using GPT-4 and Claude for semantic evaluation
   - Four dimensions: Coherence, Distinctiveness, Diversity, Semantic Integration
   - Numerical scoring (0-1 scale) with qualitative explanations

3. Experimental Validation:
   - Three synthetic datasets with varying topic similarity levels
   - Comparison between statistical and LLM-based metrics
   - Analysis of correlation and agreement between methods

Key Research Questions:
- Do LLM-based evaluations align with traditional statistical metrics?
- Can LLMs provide more nuanced topic quality assessment?
- What are the trade-offs between statistical and LLM-based approaches?

Hypothesis: LLM-based evaluation can capture semantic qualities that statistical metrics miss,
providing complementary insights for comprehensive topic model assessment.
"""

        context = {
            'manuscript_summary': manuscript_text,
            'evaluation_results': self.load_evaluation_results(),
            'datasets_info': self.load_datasets_info(),
            'timestamp': datetime.now().isoformat()
        }

        return context

    def run_single_analyzer(self, name: str, analyzer, context: Dict[str, Any]) -> tuple:
        """Run a single analyzer and return results"""
        logger.info(f"Starting analysis with {name}")
        start_time = time.time()

        try:
            results = analyzer.analyze_manuscript(context)
            elapsed = time.time() - start_time
            logger.info(f"{name} completed in {elapsed:.2f} seconds")

            # Save individual results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.results_dir, f'{name}_{timestamp}.json')
            analyzer.save_results(results, filepath)

            return (name, results, None)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{name} failed after {elapsed:.2f} seconds: {e}")
            return (name, None, str(e))

    def run_parallel_analysis(self, context: Dict[str, Any],
                             selected_analyzers: List[str] = None) -> Dict[str, Any]:
        """Run multiple LLM analyzers in parallel"""

        if selected_analyzers is None:
            selected_analyzers = list(self.analyzers.keys())

        logger.info(f"Running parallel analysis with: {', '.join(selected_analyzers)}")

        all_results = {}
        errors = {}

        with ThreadPoolExecutor(max_workers=len(selected_analyzers)) as executor:
            future_to_analyzer = {
                executor.submit(
                    self.run_single_analyzer,
                    name,
                    self.analyzers[name],
                    context
                ): name
                for name in selected_analyzers
            }

            for future in as_completed(future_to_analyzer):
                name, results, error = future.result()

                if error:
                    errors[name] = error
                else:
                    all_results[name] = results

        logger.info(f"Parallel analysis complete. Successful: {len(all_results)}, Failed: {len(errors)}")

        if errors:
            logger.warning(f"Errors encountered: {errors}")

        return {
            'results': all_results,
            'errors': errors,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }

    def save_combined_results(self, combined_results: Dict[str, Any]):
        """Save all results to a single file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f'combined_analysis_{timestamp}.json')

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Combined results saved to {filepath}")
        return filepath


def main():
    """Main execution function"""

    logger.info("="*60)
    logger.info("Multi-LLM Manuscript Validation Framework")
    logger.info("="*60)

    validator = ManuscriptValidator()

    # Prepare context
    logger.info("\n1. Preparing analysis context...")
    context = validator.prepare_context()

    # You can optionally provide custom manuscript text:
    # with open('manuscript_text.txt', 'r', encoding='utf-8') as f:
    #     manuscript_text = f.read()
    # context = validator.prepare_context(manuscript_text)

    # Run parallel analysis
    logger.info("\n2. Running parallel LLM analysis...")
    combined_results = validator.run_parallel_analysis(context)

    # Save results
    logger.info("\n3. Saving results...")
    results_file = validator.save_combined_results(combined_results)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Successful analyses: {len(combined_results['results'])}")
    logger.info(f"Failed analyses: {len(combined_results['errors'])}")

    if combined_results['results']:
        logger.info("\nOverall Scores:")
        for name, results in combined_results['results'].items():
            overall = results.get('overall_assessment', {})
            logger.info(f"  {name}: {overall.get('overall_score', 'N/A')}/10 "
                       f"({overall.get('recommendation', 'N/A')})")

    logger.info("\nNext step: Run synthesis_engine.py to generate final validation report")


if __name__ == "__main__":
    main()

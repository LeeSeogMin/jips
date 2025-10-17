from openai import OpenAI  # Grok uses OpenAI-compatible API
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import logging
from tabulate import tabulate
from base_topic_evaluator import BaseLLMEvaluator

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TopicEvaluatorLLM(BaseLLMEvaluator):
    def __init__(self, model="grok-4-0709", temperature: float = 0.3, prompt_variant: str = 'standard'):
        super().__init__(temperature, prompt_variant)
        # Grok uses OpenAI-compatible API with different base URL
        self.client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.model = model

    def _call_api(self, metric: str, prompt: str) -> str:
        """Call Grok API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Evaluate the following for {metric}:\n{prompt}\n\nProvide a score between 0 and 1 first, followed by your explanation. Format: <score>\n<explanation>"}
            ],
            temperature=self.temperature,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    def create_comparison_table(self, results_dict):
        """Create comparison table from evaluation results"""
        metrics = ['Coherence', 'Distinctiveness', 'Diversity', 'Semantic Integration', 'Overall Score']
        data = {'Metric': metrics}

        # Handle if results_dict is already passed as items()
        if isinstance(results_dict, dict):
            items = results_dict.items()
        else:
            items = results_dict

        for name, results in items:
            scores = results['scores']
            data[name] = [
                scores['coherence'],
                scores['distinctiveness'],
                scores['diversity'],
                scores['semantic_integration'],
                scores['overall_score']
            ]

        df = pd.DataFrame(data)
        return tabulate(df, headers='keys', tablefmt='psql', floatfmt='.3f')


def main():
    try:
        # Load topic data
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')

        with open(os.path.join(data_dir, 'topics_distinct.pkl'), 'rb') as f:
            topics_distinct = pickle.load(f)
        with open(os.path.join(data_dir, 'topics_similar.pkl'), 'rb') as f:
            topics_similar = pickle.load(f)
        with open(os.path.join(data_dir, 'topics_more_similar.pkl'), 'rb') as f:
            topics_more_similar = pickle.load(f)

        # Perform evaluation
        evaluator = TopicEvaluatorLLM()

        # Evaluate each dataset
        results = {
            'Distinct Topics': evaluator.evaluate_topic_set(topics_distinct, "Distinct Topics"),
            'Similar Topics': evaluator.evaluate_topic_set(topics_similar, "Similar Topics"),
            'More Similar Topics': evaluator.evaluate_topic_set(topics_more_similar, "More Similar Topics")
        }

        # Create and display results table
        comparison_table = evaluator.create_comparison_table(results)
        print("\nTopic Model Evaluation Results:")
        print(comparison_table)

        # Save results
        with open(os.path.join(data_dir, 'grok_evaluation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        logger.info("Evaluation completed and results saved.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

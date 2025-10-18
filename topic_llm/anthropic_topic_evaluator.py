from anthropic import Anthropic
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import logging
from tabulate import tabulate
from .base_topic_evaluator import BaseLLMEvaluator

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TopicEvaluatorLLM(BaseLLMEvaluator):
    def __init__(self, model="claude-sonnet-4-5-20250929", temperature: float = 0.0, prompt_variant: str = 'standard'):
        super().__init__(temperature, prompt_variant)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def _call_api(self, metric: str, prompt: str) -> str:
        """Call Anthropic API"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Evaluate the following for {metric}:\n{prompt}\n\nProvide a score between 0 and 1 first, followed by your explanation."
                }
            ]
        )

        response_text = message.content[0].text.strip()

        # Process XML tags if present
        if '<score>' in response_text and '</score>' in response_text:
            score_str = response_text.split('<score>')[1].split('</score>')[0].strip()
            explanation = response_text.split('</score>')[1].strip()
            if '<explanation>' in explanation:
                explanation = explanation.split('<explanation>')[1].split('</explanation>')[0].strip()
            return f"{score_str}\n{explanation}"

        return response_text

    def create_comparison_table(self, *results_dict) -> str:
        """Create comparison table from evaluation results"""
        metrics = ['Coherence', 'Distinctiveness', 'Diversity', 'Semantic Integration', 'Overall Score']
        data = {'Metric': metrics}
        
        for name, results in results_dict:
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
        print("\nLoading topic data...")
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
        
        print("Topic data loaded successfully.")
        
        # Perform evaluation
        evaluator = TopicEvaluatorLLM()
        
        # Evaluate each dataset
        results = {
            'Distinct Topics': evaluator.evaluate_topic_set(topics_distinct, "Distinct Topics"),
            'Similar Topics': evaluator.evaluate_topic_set(topics_similar, "Similar Topics"),
            'More Similar Topics': evaluator.evaluate_topic_set(topics_more_similar, "More Similar Topics")
        }
        
        # Create and display results table
        print("\n=== Final Comparison ===")
        comparison_table = evaluator.create_comparison_table(*results.items())
        print("\nTopic Model Evaluation Results:")
        print(comparison_table)
        
        # Save detailed results
        print("\nSaving detailed results...")
        for name, result in results.items():
            evaluator.save_detailed_results(result, os.path.join(data_dir, f'detailed_results_{name.lower().replace(" ", "_")}.txt'))

        # Save results
        with open(os.path.join(data_dir, 'anthropic_evaluation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
            
        print("\nEvaluation completed and results saved.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
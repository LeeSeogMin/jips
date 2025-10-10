import google.generativeai as genai
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import logging
from tabulate import tabulate
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_topic_evaluator import BaseLLMEvaluator
import time

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TopicEvaluatorLLM(BaseLLMEvaluator):
    def __init__(self, model="gemini-2.5-pro"):
        super().__init__()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model

        # Configure safety settings to be more permissive for academic content
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2000,
                "top_p": 1.0,
                "top_k": 40
            },
            safety_settings=safety_settings
        )

    def _call_api(self, metric: str, prompt: str) -> str:
        """Call Gemini API with retry logic and rate limiting"""
        full_prompt = f"""{self.system_prompt}

Rate this for {metric}:
{prompt}

Give a score from 0 to 1, then explain. Format: <score>
<explanation>"""

        max_retries = 3
        retry_delay = 2  # seconds
        logger.info(f"Calling Gemini API for {metric} evaluation (model: {self.model_name})")

        for attempt in range(max_retries):
            try:
                # Add delay between API calls to avoid rate limiting
                if attempt > 0:
                    time.sleep(retry_delay * attempt)

                response = self.model.generate_content(full_prompt)

                # Check if response has valid content
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    text = response.candidates[0].content.parts[0].text.strip()
                    if text:
                        return text
                
                # Log detailed error information
                logger.warning(f"Invalid response on attempt {attempt + 1}/{max_retries}")
                if hasattr(response, 'prompt_feedback'):
                    logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                if response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        logger.warning(f"Finish reason: {candidate.finish_reason}")
                        # If MAX_TOKENS, try with shorter prompt
                        if candidate.finish_reason == 2:
                            logger.warning("MAX_TOKENS reached, will retry with shorter prompt")
                continue

            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise

        raise Exception(f"Failed to get valid response after {max_retries} attempts")

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
        with open(os.path.join(data_dir, 'gemini_evaluation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        logger.info("Evaluation completed and results saved.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

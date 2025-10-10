import google.generativeai as genai
import os
from dotenv import load_dotenv
import pickle
import numpy as np
from typing import List, Tuple
import logging
import time

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeminiContinueEvaluator:
    def __init__(self, model="gemini-2.5-pro"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model

        # Simplified system prompt
        self.system_prompt = """You are a text analysis expert. Rate word groups on a scale of 0 to 1.

Score range: 0 (poor) to 1 (excellent)"""

        # Configure safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 1500,  # Reduced from 2000
                "top_p": 1.0,
                "top_k": 40
            },
            safety_settings=safety_settings
        )

    def _call_api(self, prompt: str) -> str:
        """Call Gemini API with retry logic"""
        max_retries = 3
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(retry_delay * attempt)

                response = self.model.generate_content(prompt)

                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    text = response.candidates[0].content.parts[0].text.strip()
                    if text:
                        return text

                logger.warning(f"Invalid response on attempt {attempt + 1}/{max_retries}")
                if response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        logger.warning(f"Finish reason: {candidate.finish_reason}")

            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        raise Exception(f"Failed to get valid response after {max_retries} attempts")

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """Parse LLM response to extract score and explanation"""
        import re
        try:
            # Handle XML-tagged responses
            if '<score>' in response_text and '</score>' in response_text:
                score_str = response_text.split('<score>')[1].split('</score>')[0].strip()
                score = float(score_str)

                if '<explanation>' in response_text and '</explanation>' in response_text:
                    explanation = response_text.split('<explanation>')[1].split('</explanation>')[0].strip()
                else:
                    explanation = response_text.split('</score>')[1].strip()

            # Handle bare angle bracket format
            elif re.match(r'^<(\d+\.?\d*)>\s*', response_text):
                match = re.match(r'^<(\d+\.?\d*)>\s*', response_text)
                score = float(match.group(1))
                explanation = re.sub(r'^<\d+\.?\d*>\s*', '', response_text, count=1).strip()

            # Handle markdown format
            elif re.search(r'\*\*[^*]*Score[^*]*:\s*(\d+\.?\d*)\*\*', response_text, re.IGNORECASE):
                match = re.search(r'\*\*[^*]*Score[^*]*:\s*(\d+\.?\d*)\*\*', response_text, re.IGNORECASE)
                score = float(match.group(1))
                explanation = re.sub(r'\*\*[^*]*Score[^*]*:\s*\d+\.?\d*\*\*\s*\n*', '', response_text, count=1).strip()

            else:
                # Try plain text format
                score_str, explanation = response_text.split('\n', 1)
                score = float(score_str)
                explanation = explanation.strip()

            score = min(max(score, 0), 1)
            return score, explanation
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse score from response: {response_text[:200]}...")
            return 0.5, response_text

    def evaluate_distinctiveness_short(self, all_topics: List[List[str]]) -> Tuple[float, str]:
        """Evaluate distinctiveness with shorter prompt"""
        # Create ultra-compact topic representation (first 5 keywords only)
        topics_str = "\n".join([f"T{i+1}: {', '.join(topic[:5])}" for i, topic in enumerate(all_topics)])

        prompt = f"""{self.system_prompt}

Rate distinctiveness (how different these word groups are from each other):
{topics_str}

Score 0-1, then explain briefly."""

        logger.info(f"Calling Gemini API for distinctiveness (short prompt, {len(prompt)} chars)")
        response_text = self._call_api(prompt)
        return self._parse_response(response_text)

    def evaluate_diversity_short(self, all_topics: List[List[str]]) -> Tuple[float, str]:
        """Evaluate diversity with shorter prompt"""
        topics_str = "\n".join([f"T{i+1}: {', '.join(topic[:5])}" for i, topic in enumerate(all_topics)])

        prompt = f"""{self.system_prompt}

Rate diversity (how varied these word groups are):
{topics_str}

Score 0-1, then explain briefly."""

        logger.info(f"Calling Gemini API for diversity (short prompt, {len(prompt)} chars)")
        response_text = self._call_api(prompt)
        return self._parse_response(response_text)

    def evaluate_semantic_integration_short(self, all_topics: List[List[str]]) -> Tuple[float, str]:
        """Evaluate semantic integration with shorter prompt"""
        topics_str = "\n".join([f"T{i+1}: {', '.join(topic[:5])}" for i, topic in enumerate(all_topics)])

        prompt = f"""{self.system_prompt}

Rate overall quality (coherence + distinctiveness + structure):
{topics_str}

Score 0-1, then explain briefly."""

        logger.info(f"Calling Gemini API for semantic integration (short prompt, {len(prompt)} chars)")
        response_text = self._call_api(prompt)
        return self._parse_response(response_text)


def main():
    try:
        # Load topic data
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')

        with open(os.path.join(data_dir, 'topics_distinct.pkl'), 'rb') as f:
            topics_distinct = pickle.load(f)

        # Load existing coherence results from gemini_result.md
        coherence_score = 0.975  # From gemini_result.md line 115

        evaluator = GeminiContinueEvaluator()

        print("\n=== Continuing Gemini Evaluation (Distinct Topics) ===")
        print(f"Coherence Score: {coherence_score:.3f} (already completed)")

        # Evaluate remaining metrics with shorter prompts
        print("\nEvaluating Distinctiveness (short prompt)...")
        dist_score, dist_exp = evaluator.evaluate_distinctiveness_short(topics_distinct)
        print(f"Distinctiveness Score: {dist_score:.3f}")
        print(f"Explanation: {dist_exp[:200]}...")

        print("\nEvaluating Diversity (short prompt)...")
        div_score, div_exp = evaluator.evaluate_diversity_short(topics_distinct)
        print(f"Diversity Score: {div_score:.3f}")
        print(f"Explanation: {div_exp[:200]}...")

        print("\nEvaluating Semantic Integration (short prompt)...")
        sem_score, sem_exp = evaluator.evaluate_semantic_integration_short(topics_distinct)
        print(f"Semantic Integration Score: {sem_score:.3f}")
        print(f"Explanation: {sem_exp[:200]}...")

        # Calculate overall score
        overall_score = (
            coherence_score * 0.3 +
            dist_score * 0.3 +
            div_score * 0.2 +
            sem_score * 0.2
        )

        print(f"\nOverall Score: {overall_score:.3f}")
        print("=" * 50)

        # Save partial results
        results = {
            'Distinct Topics': {
                'scores': {
                    'coherence': coherence_score,
                    'distinctiveness': dist_score,
                    'diversity': div_score,
                    'semantic_integration': sem_score,
                    'overall_score': overall_score
                },
                'explanations': {
                    'coherence': "Completed in previous run (see gemini_result.md)",
                    'distinctiveness': dist_exp,
                    'diversity': div_exp,
                    'semantic_integration': sem_exp
                }
            }
        }

        with open(os.path.join(data_dir, 'gemini_partial_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✅ Partial results saved to: gemini_partial_results.pkl")
        logger.info("Partial evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()

from openai import OpenAI
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import logging
from tabulate import tabulate

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TopicEvaluatorLLM:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = """You are an expert in topic modeling evaluation. Your role is to evaluate topic models based on four key metrics:

1. Coherence: Assess how semantically coherent and meaningful the keywords within each topic are.
2. Distinctiveness: Evaluate how well-differentiated and unique each topic is from others.
3. Diversity: Analyze both semantic diversity (meaning variation) and distribution diversity (balanced coverage).
4. Semantic Integration: Provide a holistic evaluation combining coherence, distinctiveness, and overall topic structure.

Provide numerical scores between 0 and 1, where:
- 0: Poor performance on the metric
- 0.5: Average performance
- 1: Excellent performance

Base your evaluation on academic standards and best practices in topic modeling."""

    def _get_llm_score(self, metric: str, prompt: str) -> tuple[float, str]:
        """Get evaluation score and explanation using LLM"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Evaluate the following for {metric}:\n{prompt}\n\nProvide a score between 0 and 1 first, followed by your explanation. Format: <score>\n<explanation>"}
                ],
                temperature=0,
                max_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Separate score and explanation from response
            try:
                score_str, explanation = response_text.split('\n', 1)
                score = float(score_str)
                score = min(max(score, 0), 1)  # Limit score range
                return score, explanation.strip()
            except ValueError:
                logger.error(f"Failed to parse score from response: {response_text}")
                return 0.5, response_text
                
        except Exception as e:
            logger.error(f"Error in LLM API call for {metric}: {e}")
            return 0.5, f"Error occurred: {str(e)}"

    def evaluate_coherence(self, keywords: List[str]) -> float:
        """Evaluate semantic coherence of keywords within a topic"""
        prompt = f"""Evaluate the semantic coherence of these keywords:
Keywords: {', '.join(keywords)}

Consider:
1. Semantic similarity between keywords
2. Logical relationship and theme consistency
3. Absence of outlier or unrelated terms
4. Clear thematic focus"""
        
        return self._get_llm_score("coherence", prompt)

    def evaluate_distinctiveness(self, topic1: List[str], topic2: List[str]) -> float:
        """Evaluate distinctiveness between topics"""
        prompt = f"""Compare these two topics for distinctiveness:
Topic 1: {', '.join(topic1)}
Topic 2: {', '.join(topic2)}

Consider:
1. Semantic overlap between topics
2. Unique thematic focus of each topic
3. Clarity of boundaries between topics
4. Potential confusion or ambiguity"""
        
        return self._get_llm_score("distinctiveness", prompt)

    def evaluate_diversity(self, all_topics: List[List[str]]) -> float:
        """Evaluate semantic diversity of the topic set"""
        topics_str = "\n".join([f"Topic {i+1}: {', '.join(topic)}" for i, topic in enumerate(all_topics)])
        prompt = f"""Evaluate the overall diversity of this topic set:
{topics_str}

Consider:
1. Coverage of different themes and concepts
2. Balance in topic distribution
3. Semantic range and variation
4. Absence of redundant or overlapping topics"""
        
        return self._get_llm_score("diversity", prompt)

    def evaluate_semantic_integration(self, all_topics: List[List[str]]) -> float:
        """Evaluate overall semantic integration of the topic model"""
        topics_str = "\n".join([f"Topic {i+1}: {', '.join(topic)}" for i, topic in enumerate(all_topics)])
        prompt = f"""Evaluate the overall semantic integration of this topic model:
{topics_str}

Consider:
1. Overall topic model coherence
2. Balance between distinctiveness and relationships
3. Hierarchical topic structure
4. Practical interpretability and usefulness"""
        
        return self._get_llm_score("semantic_integration", prompt)

    def evaluate_topic_set(self, topic_keywords: List[List[str]], set_name: str) -> Dict[str, Union[float, Dict[str, str]]]:
        """Perform comprehensive evaluation on a topic set"""
        print(f"\n=== Evaluating {set_name} ===")
        results = {
            'scores': {},
            'explanations': {}
        }
        
        # 1. Evaluate coherence for each topic
        print("\nEvaluating Coherence...")
        coherence_scores = []
        coherence_explanations = []
        for i, keywords in enumerate(topic_keywords, 1):
            score, explanation = self.evaluate_coherence(keywords)
            print(f"Topic {i}: {score:.3f}")
            print(f"Explanation: {explanation}\n")
            coherence_scores.append(score)
            coherence_explanations.append(explanation)
        results['scores']['coherence'] = np.mean(coherence_scores)
        results['explanations']['coherence'] = coherence_explanations
        print(f"Average Coherence Score: {results['scores']['coherence']:.3f}")
        
        # 2. Evaluate distinctiveness between topics
        print("\nEvaluating Distinctiveness...")
        distinctiveness_scores = []
        distinctiveness_explanations = []
        for i in range(len(topic_keywords)):
            for j in range(i+1, len(topic_keywords)):
                score, explanation = self.evaluate_distinctiveness(topic_keywords[i], topic_keywords[j])
                print(f"Topics {i+1} vs {j+1}: {score:.3f}")
                print(f"Explanation: {explanation}\n")
                distinctiveness_scores.append(score)
                distinctiveness_explanations.append(explanation)
        results['scores']['distinctiveness'] = np.mean(distinctiveness_scores)
        results['explanations']['distinctiveness'] = distinctiveness_explanations
        print(f"Average Distinctiveness Score: {results['scores']['distinctiveness']:.3f}")
        
        # 3. Evaluate topic diversity
        print("\nEvaluating Diversity...")
        score, explanation = self.evaluate_diversity(topic_keywords)
        print(f"Diversity Score: {score:.3f}")
        print(f"Explanation: {explanation}\n")
        results['scores']['diversity'] = score
        results['explanations']['diversity'] = explanation
        
        # 4. Evaluate semantic integration
        print("\nEvaluating Semantic Integration...")
        score, explanation = self.evaluate_semantic_integration(topic_keywords)
        print(f"Semantic Integration Score: {score:.3f}")
        print(f"Explanation: {explanation}\n")
        results['scores']['semantic_integration'] = score
        results['explanations']['semantic_integration'] = explanation
        
        # Calculate overall score
        results['scores']['overall_score'] = (
            results['scores']['coherence'] * 0.3 +
            results['scores']['distinctiveness'] * 0.3 +
            results['scores']['diversity'] * 0.2 +
            results['scores']['semantic_integration'] * 0.2
        )
        print(f"Overall Score: {results['scores']['overall_score']:.3f}")
        print("=" * 50)
        
        return results

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

    def save_detailed_results(self, results: Dict[str, Dict[str, Union[float, Dict[str, str]]]], filename: str):
        """Save detailed evaluation results to a text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Detailed Evaluation Results ===\n\n")
            
            for metric in ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']:
                f.write(f"\n=== {metric.upper()} ===\n")
                f.write(f"Score: {results['scores'][metric]:.3f}\n")
                f.write("Explanations:\n")
                
                explanations = results['explanations'][metric]
                if isinstance(explanations, list):
                    for i, exp in enumerate(explanations, 1):
                        f.write(f"{i}. {exp}\n")
                else:
                    f.write(explanations + "\n")
            
            f.write(f"\nOverall Score: {results['scores']['overall_score']:.3f}\n")

def main():
    try:
        # Load topic data
        with open('data/topics_distinct.pkl', 'rb') as f:
            topics_distinct = pickle.load(f)
        with open('data/topics_similar.pkl', 'rb') as f:
            topics_similar = pickle.load(f)
        with open('data/topics_more_similar.pkl', 'rb') as f:
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
        with open('data/llm_evaluation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        logger.info("Evaluation completed and results saved.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import logging
from tabulate import tabulate

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TopicEvaluatorLLM:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.system_prompt = """You are an expert in topic modeling evaluation. Your role is to evaluate topic models based on four key metrics:

1. Coherence: Assess how semantically coherent and meaningful the keywords within each topic are.
2. Distinctiveness: Evaluate how well-differentiated and unique each topic is from others.
3. Diversity: Analyze both semantic diversity (meaning variation) and distribution diversity (balanced coverage).
4. Semantic Integration: Provide a holistic evaluation combining coherence, distinctiveness, and overall topic structure.

Provide numerical scores between 0 and 1, where:
- 0: Poor performance on the metric
- 0.5: Average performance
- 1: Excellent performance

Base your evaluation on academic standards and best practices in topic modeling.

Important: Always format your response as:
<score>
<explanation>"""

    def _get_llm_score(self, metric: str, prompt: str) -> Tuple[float, str]:
        """Get evaluation score and explanation using LLM"""
        try:
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=150,
                temperature=0,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Evaluate the following for {metric}:\n{prompt}\n\nProvide a score between 0 and 1 first, followed by your explanation."
                    }
                ]
            )
            
            response_text = message.content[0].text.strip()
            
            # Process XML tags
            if '<score>' in response_text and '</score>' in response_text:
                score_str = response_text.split('<score>')[1].split('</score>')[0].strip()
                explanation = response_text.split('</score>')[1].strip()
                if '<explanation>' in explanation:
                    explanation = explanation.split('<explanation>')[1].split('</explanation>')[0].strip()
            else:
                # Maintain existing processing method
                score_str, explanation = response_text.split('\n', 1)
                
            score = float(score_str)
            score = min(max(score, 0), 1)  # Limit score range
            return score, explanation.strip()
                
        except Exception as e:
            logger.error(f"Error in LLM API call for {metric}: {e}")
            return 0.5, f"Error occurred: {str(e)}"

    def evaluate_coherence(self, keywords: List[str]) -> Tuple[float, str]:
        """Evaluate semantic coherence of keywords within a topic"""
        prompt = f"""Evaluate the semantic coherence of these keywords:
Keywords: {', '.join(keywords)}

Consider:
1. Semantic similarity between keywords
2. Logical relationship and theme consistency
3. Absence of outlier or unrelated terms
4. Clear thematic focus"""
        
        return self._get_llm_score("coherence", prompt)

    def evaluate_distinctiveness(self, topic1: List[str], topic2: List[str]) -> Tuple[float, str]:
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

    def evaluate_diversity(self, all_topics: List[List[str]]) -> Tuple[float, str]:
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

    def evaluate_semantic_integration(self, all_topics: List[List[str]]) -> Tuple[float, str]:
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
        print("\n[Evaluating Coherence]")
        coherence_scores = []
        coherence_explanations = []
        for i, keywords in enumerate(topic_keywords, 1):
            score, explanation = self.evaluate_coherence(keywords)
            print(f"\nTopic {i}:")
            print(f"Score: {score:.3f}")
            print(f"Explanation: {explanation}")
            coherence_scores.append(score)
            coherence_explanations.append(explanation)
        results['scores']['coherence'] = np.mean(coherence_scores)
        results['explanations']['coherence'] = coherence_explanations
        print(f"\nAverage Coherence Score: {results['scores']['coherence']:.3f}")
        
        # 2. Evaluate distinctiveness between topics
        print("\n[Evaluating Distinctiveness]")
        distinctiveness_scores = []
        distinctiveness_explanations = []
        for i in range(len(topic_keywords)):
            for j in range(i+1, len(topic_keywords)):
                score, explanation = self.evaluate_distinctiveness(topic_keywords[i], topic_keywords[j])
                print(f"\nComparing Topic {i+1} vs Topic {j+1}:")
                print(f"Score: {score:.3f}")
                print(f"Explanation: {explanation}")
                distinctiveness_scores.append(score)
                distinctiveness_explanations.append(explanation)
        results['scores']['distinctiveness'] = np.mean(distinctiveness_scores)
        results['explanations']['distinctiveness'] = distinctiveness_explanations
        print(f"\nAverage Distinctiveness Score: {results['scores']['distinctiveness']:.3f}")
        
        # 3. Evaluate topic diversity
        print("\n[Evaluating Diversity]")
        score, explanation = self.evaluate_diversity(topic_keywords)
        print(f"\nScore: {score:.3f}")
        print(f"Explanation: {explanation}")
        results['scores']['diversity'] = score
        results['explanations']['diversity'] = explanation
        
        # 4. Evaluate semantic integration
        print("\n[Evaluating Semantic Integration]")
        score, explanation = self.evaluate_semantic_integration(topic_keywords)
        print(f"\nScore: {score:.3f}")
        print(f"Explanation: {explanation}")
        results['scores']['semantic_integration'] = score
        results['explanations']['semantic_integration'] = explanation
        
        # Calculate overall score
        results['scores']['overall_score'] = (
            results['scores']['coherence'] * 0.3 +
            results['scores']['distinctiveness'] * 0.3 +
            results['scores']['diversity'] * 0.2 +
            results['scores']['semantic_integration'] * 0.2
        )
        print(f"\nOverall Score: {results['scores']['overall_score']:.3f}")
        
        return results

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
        print("\nLoading topic data...")
        # Load topic data
        with open('data/topics_distinct.pkl', 'rb') as f:
            topics_distinct = pickle.load(f)
        with open('data/topics_similar.pkl', 'rb') as f:
            topics_similar = pickle.load(f)
        with open('data/topics_more_similar.pkl', 'rb') as f:
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
            evaluator.save_detailed_results(result, f'data/detailed_results_{name.lower().replace(" ", "_")}.txt')
        
        # Save results
        with open('data/claude_evaluation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        print("\nEvaluation completed and results saved.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
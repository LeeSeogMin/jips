from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Union, Tuple
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMEvaluator(ABC):
    """Base class for LLM-based topic model evaluation"""

    def __init__(self, temperature: float = 0.3, prompt_variant: str = 'standard'):
        self.temperature = temperature
        self.prompt_variant = prompt_variant
        
        # Define different prompt variants
        self.prompt_variants = {
            'standard': """You are a text analysis expert. Rate word groups on a scale of 0 to 1.

Provide scores for:
1. Coherence: How well words relate to each other
2. Distinctiveness: How different word groups are from each other
3. Diversity: How varied the word groups are
4. Integration: Overall quality of the word groups

Score range: 0 (poor) to 1 (excellent)""",
            
            'detailed': """You are an expert in computational linguistics and topic modeling. Evaluate the following word groups with precision.

For each metric, provide a score from 0.0 to 1.0:

1. COHERENCE: Assess semantic relatedness and thematic consistency
   - High coherence: Words form a clear, interpretable theme
   - Low coherence: Words are unrelated or contradictory

2. DISTINCTIVENESS: Measure how well this group differs from others
   - High distinctiveness: Unique, separable concepts
   - Low distinctiveness: Overlapping or generic terms

3. DIVERSITY: Evaluate lexical and semantic variety
   - High diversity: Rich, varied vocabulary
   - Low diversity: Repetitive or narrow terms

4. INTEGRATION: Overall quality and interpretability
   - High integration: Cohesive, meaningful topic
   - Low integration: Fragmented or unclear grouping

Provide scores as: <score>X.XX</score> <explanation>Brief rationale</explanation>""",
            
            'concise': """Rate word groups (0-1 scale):

1. Coherence: Word relatedness
2. Distinctiveness: Uniqueness vs other groups  
3. Diversity: Vocabulary variety
4. Integration: Overall quality

Format: <score>X.XX</score> <explanation>Brief reason</explanation>"""
        }
        
        self.system_prompt = self.prompt_variants.get(prompt_variant, self.prompt_variants['standard'])

    @abstractmethod
    def _call_api(self, metric: str, prompt: str) -> str:
        """Call the specific LLM API - must be implemented by subclasses"""
        pass

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """Parse LLM response to extract score and explanation"""
        import re
        try:
            # Handle XML-tagged responses (e.g., <score>0.95</score>)
            if '<score>' in response_text and '</score>' in response_text:
                score_str = response_text.split('<score>')[1].split('</score>')[0].strip()
                score = float(score_str)

                # Extract explanation
                if '<explanation>' in response_text and '</explanation>' in response_text:
                    explanation = response_text.split('<explanation>')[1].split('</explanation>')[0].strip()
                else:
                    # Explanation might be after </score> without tags
                    explanation = response_text.split('</score>')[1].strip()
                    if explanation.startswith('<explanation>'):
                        explanation = explanation.split('<explanation>')[1].split('</explanation>')[0].strip()

            # Handle bare angle bracket format (e.g., <0.95> or <1.0>)
            elif re.match(r'^<(\d+\.?\d*)>\s*', response_text):
                match = re.match(r'^<(\d+\.?\d*)>\s*', response_text)
                score = float(match.group(1))
                # Rest is explanation
                explanation = re.sub(r'^<\d+\.?\d*>\s*', '', response_text, count=1).strip()
                # Remove <explanation> tags if present
                if explanation.startswith('<explanation>'):
                    explanation = explanation.split('<explanation>')[1].split('</explanation>')[0].strip()

            # Handle markdown format (e.g., **Score: 0.92**)
            elif re.search(r'\*\*[^*]*Score[^*]*:\s*(\d+\.?\d*)\*\*', response_text, re.IGNORECASE):
                match = re.search(r'\*\*[^*]*Score[^*]*:\s*(\d+\.?\d*)\*\*', response_text, re.IGNORECASE)
                score = float(match.group(1))
                # Remove the score line from explanation
                explanation = re.sub(r'\*\*[^*]*Score[^*]*:\s*\d+\.?\d*\*\*\s*\n*', '', response_text, count=1).strip()

            else:
                # Try plain text format (score\nexplanation)
                score_str, explanation = response_text.split('\n', 1)
                score = float(score_str)
                explanation = explanation.strip()

            score = min(max(score, 0), 1)  # Limit score range to [0, 1]
            return score, explanation
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse score from response: {response_text[:200]}...")
            return 0.5, response_text

    def _get_llm_score(self, metric: str, prompt: str) -> Tuple[float, str]:
        """Get evaluation score and explanation using LLM"""
        try:
            response_text = self._call_api(metric, prompt)
            return self._parse_response(response_text)
        except Exception as e:
            logger.error(f"Error in LLM API call for {metric}: {e}")
            raise Exception(f"Failed to evaluate {metric}: {str(e)}")

    def evaluate_coherence(self, keywords: List[str]) -> Tuple[float, str]:
        """Evaluate semantic coherence of keywords within a topic"""
        prompt = f"""Rate the coherence of these words: {', '.join(keywords)}

Consider:
1. How well the words relate to each other
2. Whether they form a clear theme
3. If any words seem unrelated
4. Overall consistency"""

        return self._get_llm_score("coherence", prompt)

    def evaluate_distinctiveness(self, topic1: List[str], topic2: List[str]) -> Tuple[float, str]:
        """Evaluate distinctiveness between two topics (legacy method)"""
        prompt = f"""Compare these two topics for distinctiveness:
Topic 1: {', '.join(topic1)}
Topic 2: {', '.join(topic2)}

Consider:
1. Semantic overlap between topics
2. Unique thematic focus of each topic
3. Clarity of boundaries between topics
4. Potential confusion or ambiguity"""

        return self._get_llm_score("distinctiveness", prompt)

    def evaluate_distinctiveness_aggregated(self, all_topics: List[List[str]]) -> Tuple[float, str]:
        """Evaluate overall distinctiveness of the topic set (manuscript method)"""
        topics_str = "\n".join([f"Topic {i+1}: {', '.join(topic)}" for i, topic in enumerate(all_topics)])
        prompt = f"""Evaluate how well-differentiated these topics are from each other:
{topics_str}

Consider:
1. Semantic overlap between topics
2. Unique thematic focus of each topic
3. Clarity of boundaries between topics
4. Overall topic separation quality

Provide a score between 0 and 1 where:
- 1.0: Topics are completely distinct with no overlap
- 0.5: Moderate distinction with some overlap
- 0.0: Topics are highly similar or redundant"""

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

        # 2. Evaluate distinctiveness (aggregated method - manuscript approach)
        print("\nEvaluating Distinctiveness...")
        score, explanation = self.evaluate_distinctiveness_aggregated(topic_keywords)
        print(f"Distinctiveness Score: {score:.3f}")
        print(f"Explanation: {explanation}\n")
        results['scores']['distinctiveness'] = score
        results['explanations']['distinctiveness'] = explanation

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

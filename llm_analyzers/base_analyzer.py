"""
Base class for LLM-based research analyzers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseResearchAnalyzer(ABC):
    """Abstract base class for LLM research analyzers"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.analysis_dimensions = [
            "methodology_validity",
            "statistical_rigor",
            "experimental_design",
            "results_interpretation",
            "contribution_assessment",
            "limitations_identification"
        ]

    @abstractmethod
    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Call the LLM API with given prompt

        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            LLM response text
        """
        pass

    def analyze_dimension(self, dimension: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a specific research dimension

        Args:
            dimension: One of the analysis dimensions
            context: Dictionary containing manuscript text, evaluation results, etc.

        Returns:
            Analysis result with score, reasoning, and recommendations
        """
        from validation_prompts import get_prompt_for_dimension

        prompt = get_prompt_for_dimension(dimension, context)

        try:
            response = self._call_llm(prompt, temperature=0.0)
            result = self._parse_response(response, dimension)
            return result
        except Exception as e:
            logger.error(f"{self.model_name} - Error analyzing {dimension}: {e}")
            return {
                "dimension": dimension,
                "score": None,
                "reasoning": f"Error occurred: {str(e)}",
                "recommendations": [],
                "confidence": 0.0
            }

    def _parse_response(self, response: str, dimension: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format
        Expected format:
        SCORE: <0-10>
        CONFIDENCE: <0-1>
        REASONING: <text>
        RECOMMENDATIONS: <list>
        """
        result = {
            "dimension": dimension,
            "score": None,
            "reasoning": "",
            "recommendations": [],
            "confidence": 0.0,
            "raw_response": response
        }

        try:
            lines = response.strip().split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith("SCORE:"):
                    try:
                        result["score"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        result["score"] = 5.0
                elif line.startswith("CONFIDENCE:"):
                    try:
                        result["confidence"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        result["confidence"] = 0.5
                elif line.startswith("REASONING:"):
                    current_section = "reasoning"
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("RECOMMENDATIONS:"):
                    current_section = "recommendations"
                elif line.startswith("-") and current_section == "recommendations":
                    result["recommendations"].append(line[1:].strip())
                elif current_section == "reasoning" and line:
                    result["reasoning"] += " " + line
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")

        return result

    def analyze_manuscript(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive manuscript analysis across all dimensions

        Args:
            context: Complete research context including manuscript and results

        Returns:
            Complete analysis results
        """
        logger.info(f"{self.model_name} - Starting manuscript analysis")

        results = {
            "model": self.model_name,
            "analyses": {},
            "overall_assessment": None
        }

        # Analyze each dimension
        for dimension in self.analysis_dimensions:
            logger.info(f"{self.model_name} - Analyzing: {dimension}")
            analysis = self.analyze_dimension(dimension, context)
            results["analyses"][dimension] = analysis

        # Generate overall assessment
        logger.info(f"{self.model_name} - Generating overall assessment")
        results["overall_assessment"] = self._generate_overall_assessment(results["analyses"])

        return results

    def _generate_overall_assessment(self, analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall assessment from dimension analyses"""
        scores = [a["score"] for a in analyses.values() if a["score"] is not None]

        if not scores:
            return {
                "overall_score": None,
                "recommendation": "Unable to assess",
                "key_strengths": [],
                "key_weaknesses": [],
                "publication_readiness": "UNCERTAIN"
            }

        avg_score = sum(scores) / len(scores)

        # Determine publication readiness
        if avg_score >= 8.0:
            readiness = "READY"
        elif avg_score >= 6.0:
            readiness = "MINOR_REVISIONS"
        elif avg_score >= 4.0:
            readiness = "MAJOR_REVISIONS"
        else:
            readiness = "REJECT"

        # Extract key points
        strengths = []
        weaknesses = []

        for dim, analysis in analyses.items():
            if analysis["score"] and analysis["score"] >= 7.0:
                strengths.append(f"{dim}: {analysis['reasoning'][:100]}...")
            elif analysis["score"] and analysis["score"] < 5.0:
                weaknesses.append(f"{dim}: {analysis['reasoning'][:100]}...")

        return {
            "overall_score": round(avg_score, 2),
            "recommendation": readiness,
            "key_strengths": strengths[:3],
            "key_weaknesses": weaknesses[:3],
            "avg_confidence": round(sum(a["confidence"] for a in analyses.values()) / len(analyses), 2)
        }

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save analysis results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"{self.model_name} - Results saved to {filepath}")

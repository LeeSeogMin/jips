"""
OpenAI GPT-4.1 based research analyzer
Updated: 2025-10 with latest GPT-4.1 model
"""

from .base_analyzer import BaseResearchAnalyzer
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class OpenAIAnalyzer(BaseResearchAnalyzer):
    """OpenAI GPT-4.1 analyzer implementation (Latest 2025)"""

    def __init__(self, model="gpt-4.1"):
        super().__init__(model_name=f"OpenAI-{model}")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert peer reviewer for academic research in natural language processing and topic modeling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

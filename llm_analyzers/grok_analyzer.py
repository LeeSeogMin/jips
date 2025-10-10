"""
xAI Grok 4 based research analyzer
Updated: 2025-10 with latest Grok 4 model
"""

from .base_analyzer import BaseResearchAnalyzer
from openai import OpenAI  # Grok uses OpenAI-compatible API
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class GrokAnalyzer(BaseResearchAnalyzer):
    """xAI Grok 4 analyzer implementation (Latest 2025)"""

    def __init__(self, model="grok-4-0709"):
        super().__init__(model_name=f"Grok-{model}")
        # Grok uses OpenAI-compatible API with different base URL
        self.client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """Call Grok API"""
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
            logger.error(f"Grok API error: {e}")
            raise

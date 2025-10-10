"""
Anthropic Claude Sonnet 4.5 based research analyzer
Updated: 2025-10 with latest Claude Sonnet 4.5 model
"""

from .base_analyzer import BaseResearchAnalyzer
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class AnthropicAnalyzer(BaseResearchAnalyzer):
    """Anthropic Claude Sonnet 4.5 analyzer implementation (Latest 2025)"""

    def __init__(self, model="claude-sonnet-4-5-20250929"):
        super().__init__(model_name=f"Anthropic-{model}")
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """Call Anthropic API"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=temperature,
                system="You are an expert peer reviewer for academic research in natural language processing and topic modeling.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

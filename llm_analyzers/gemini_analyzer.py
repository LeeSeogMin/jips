"""
Google Gemini 2.5 Flash based research analyzer
Updated: 2025-10 with latest Gemini 2.5 Flash model
"""

from .base_analyzer import BaseResearchAnalyzer
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiAnalyzer(BaseResearchAnalyzer):
    """Google Gemini 2.5 Flash analyzer implementation (Latest 2025)"""

    def __init__(self, model="gemini-2.5-flash-preview-09-2025"):
        super().__init__(model_name=f"Gemini-{model}")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2000,
            },
            system_instruction="You are an expert peer reviewer for academic research in natural language processing and topic modeling."
        )

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """Call Gemini API"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

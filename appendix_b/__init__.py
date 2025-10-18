"""
Appendix B Validation Package

Validates manuscript Tables B1 and B2 through experimental testing
of temperature sensitivity and prompt variation robustness.
"""

from .temperature_analysis import TemperatureValidator
from .prompt_variation_analysis import PromptValidator
from .comparison_report import generate_comparison_report

__all__ = [
    'TemperatureValidator',
    'PromptValidator',
    'generate_comparison_report'
]

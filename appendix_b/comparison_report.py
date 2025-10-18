"""
Comparison Report Generator for Appendix B Validation

Compares experimental validation results against manuscript Tables B1 and B2
to identify discrepancies and generate validation report.
"""

from typing import Dict
import numpy as np
from pathlib import Path


def parse_manuscript_table_b1(manuscript_path: str) -> Dict:
    """
    Extract expected values from manuscript Table B1

    Returns:
        {
            0.0: {'coherence': {'mean': 0.920, 'std': 0.018, 'cv': 2.8}, ...},
            0.3: {...},
            0.5: {...},
            0.7: {...}
        }
    """
    # Hard-coded expected values from manuscript_revision12.md Table B1
    return {
        0.0: {
            'coherence': {'mean': 0.920, 'std': 0.018, 'cv': 2.8},
            'distinctiveness': {'mean': 0.720, 'std': 0.025, 'cv': 3.5},
            'diversity': {'mean': 0.620, 'std': 0.042, 'cv': 6.8},
            'semantic_integration': {'mean': 0.820, 'std': 0.021, 'cv': 2.6},
            'overall': {'mean': 0.780, 'std': 0.019, 'cv': 2.4},
            'mean_cv': 2.8
        },
        0.3: {
            'coherence': {'mean': 0.915, 'std': 0.024, 'cv': 2.6},
            'distinctiveness': {'mean': 0.715, 'std': 0.031, 'cv': 4.3},
            'diversity': {'mean': 0.625, 'std': 0.051, 'cv': 8.2},
            'semantic_integration': {'mean': 0.815, 'std': 0.028, 'cv': 3.4},
            'overall': {'mean': 0.775, 'std': 0.025, 'cv': 3.2},
            'mean_cv': 3.6
        },
        0.5: {
            'coherence': {'mean': 0.905, 'std': 0.035, 'cv': 3.9},
            'distinctiveness': {'mean': 0.705, 'std': 0.042, 'cv': 6.0},
            'diversity': {'mean': 0.630, 'std': 0.068, 'cv': 10.8},
            'semantic_integration': {'mean': 0.805, 'std': 0.038, 'cv': 4.7},
            'overall': {'mean': 0.765, 'std': 0.036, 'cv': 4.7},
            'mean_cv': 5.2
        },
        0.7: {
            'coherence': {'mean': 0.895, 'std': 0.051, 'cv': 5.7},
            'distinctiveness': {'mean': 0.695, 'std': 0.058, 'cv': 8.3},
            'diversity': {'mean': 0.635, 'std': 0.095, 'cv': 15.0},
            'semantic_integration': {'mean': 0.790, 'std': 0.054, 'cv': 6.8},
            'overall': {'mean': 0.755, 'std': 0.051, 'cv': 6.8},
            'mean_cv': 7.4
        }
    }


def parse_manuscript_table_b2(manuscript_path: str) -> Dict:
    """
    Extract expected values from manuscript Table B2

    Returns:
        {
            'mean_cv': 1.9,
            'classification': 'VERY LOW'
        }
    """
    # Hard-coded expected values from manuscript_revision12.md Table B2
    return {
        'mean_cv': 1.9,
        'classification': 'VERY LOW'
    }


def compare_temperature_results(experimental: Dict, expected: Dict) -> str:
    """
    Compare experimental temperature results with manuscript Table B1

    Args:
        experimental: Results from TemperatureValidator.validate()
        expected: Expected values from manuscript

    Returns:
        Markdown comparison report
    """
    lines = []
    lines.append("## Temperature Sensitivity Analysis (B.1) Validation")
    lines.append("")

    metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']

    for temp in experimental['temperatures']:
        lines.append(f"### Temperature T={temp}")
        lines.append("")
        lines.append("| Metric | Expected Mean | Experimental Mean | Δ | Expected CV | Experimental CV | Δ | Status |")
        lines.append("|--------|---------------|-------------------|---|-------------|-----------------|---|--------|")

        exp_summary = experimental['summary'][temp]
        exp_expected = expected[temp]

        for metric in metrics:
            exp_mean = exp_summary[metric]['mean']
            # Use cross_temp_cv (variation across temperatures) for reviewer request
            exp_cv = exp_summary[metric].get('cross_temp_cv', exp_summary[metric].get('within_cv', 0.0))
            expected_mean = exp_expected[metric]['mean']
            expected_cv = exp_expected[metric]['cv']

            delta_mean = abs(exp_mean - expected_mean)
            delta_cv = abs(exp_cv - expected_cv)

            # Validation thresholds
            mean_threshold = 0.05  # 5% difference allowed
            cv_threshold = 1.0  # 1% CV difference allowed

            status = "✅" if (delta_mean <= mean_threshold and delta_cv <= cv_threshold) else "❌"

            lines.append(f"| {metric.capitalize()} | {expected_mean:.3f} | {exp_mean:.3f} | {delta_mean:.3f} | {expected_cv:.1f}% | {exp_cv:.1f}% | {delta_cv:.1f}% | {status} |")

        lines.append("")

    return '\n'.join(lines)


def compare_prompt_results(experimental: Dict, expected: Dict) -> str:
    """
    Compare experimental prompt variation results with manuscript Table B2

    Args:
        experimental: Results from PromptValidator.validate()
        expected: Expected values from manuscript

    Returns:
        Markdown comparison report
    """
    lines = []
    lines.append("## Prompt Variation Analysis (B.2) Validation")
    lines.append("")

    exp_cv = experimental['summary']['mean_cv']
    expected_cv = expected['mean_cv']
    delta_cv = abs(exp_cv - expected_cv)

    lines.append("| Metric | Expected | Experimental | Δ | Status |")
    lines.append("|--------|----------|--------------|---|--------|")

    cv_threshold = 0.5  # 0.5% CV difference allowed
    status = "✅" if delta_cv <= cv_threshold else "❌"

    lines.append(f"| Mean CV | {expected_cv:.1f}% | {exp_cv:.1f}% | {delta_cv:.1f}% | {status} |")
    lines.append(f"| Classification | {expected['classification']} | {experimental['summary']['classification']} | - | - |")
    lines.append("")

    return '\n'.join(lines)


def generate_comparison_report(
    temp_results: Dict,
    prompt_results: Dict,
    manuscript_path: str
) -> str:
    """
    Generate comprehensive validation report comparing experimental results
    with manuscript Tables B1 and B2

    Args:
        temp_results: Temperature validation results
        prompt_results: Prompt variation validation results
        manuscript_path: Path to manuscript file

    Returns:
        Markdown validation report
    """
    lines = []

    # Header
    lines.append("# Appendix B Validation Report")
    lines.append("")
    lines.append("This report compares experimental validation results against")
    lines.append("manuscript Tables B1 and B2 to verify numerical accuracy.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Parse expected values
    expected_b1 = parse_manuscript_table_b1(manuscript_path)
    expected_b2 = parse_manuscript_table_b2(manuscript_path)

    # Temperature comparison
    temp_comparison = compare_temperature_results(temp_results, expected_b1)
    lines.append(temp_comparison)
    lines.append("")

    # Prompt comparison
    prompt_comparison = compare_prompt_results(prompt_results, expected_b2)
    lines.append(prompt_comparison)
    lines.append("")

    # Overall validation status
    lines.append("---")
    lines.append("")
    lines.append("## Overall Validation Status")
    lines.append("")

    # Check if all validations passed
    all_passed = True

    # Check temperature validations
    for temp in temp_results['temperatures']:
        exp_summary = temp_results['summary'][temp]
        exp_expected = expected_b1[temp]

        for metric in ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']:
            delta_mean = abs(exp_summary[metric]['mean'] - exp_expected[metric]['mean'])
            # Use cross_temp_cv for cross-temperature variation
            exp_cv = exp_summary[metric].get('cross_temp_cv', exp_summary[metric].get('within_cv', 0.0))
            delta_cv = abs(exp_cv - exp_expected[metric]['cv'])

            if delta_mean > 0.05 or delta_cv > 1.0:
                all_passed = False
                break

    # Check prompt validation
    delta_cv = abs(prompt_results['summary']['mean_cv'] - expected_b2['mean_cv'])
    if delta_cv > 0.5:
        all_passed = False

    if all_passed:
        lines.append("✅ **PASSED**: All experimental results match manuscript claims within tolerance.")
        lines.append("")
        lines.append("**Recommendation**: Manuscript Tables B1 and B2 are validated. No revisions needed.")
    else:
        lines.append("❌ **FAILED**: Some experimental results deviate from manuscript claims.")
        lines.append("")
        lines.append("**Recommendation**: Review discrepancies and update manuscript tables with experimental results.")

    lines.append("")

    # Metadata
    lines.append("---")
    lines.append("")
    lines.append("## Validation Metadata")
    lines.append("")
    lines.append(f"- **Temperature Analysis**: {len(temp_results['temperatures'])} temperatures, {temp_results['summary'][0.0]['coherence']['mean']} topics")
    lines.append(f"- **Prompt Analysis**: {len(prompt_results['variants'])} variants, {prompt_results['topics']} topics")
    lines.append(f"- **Total API Calls**: {len(temp_results['temperatures']) * 3 * prompt_results['topics'] * 4 + len(prompt_results['variants']) * prompt_results['topics']}")
    lines.append("")

    return '\n'.join(lines)


if __name__ == '__main__':
    # Test with dummy data
    print("Comparison Report Generator - Test Mode")
    print("=" * 60)

    # Create dummy experimental results matching expected values
    temp_results = {
        'temperatures': [0.0, 0.3, 0.5, 0.7],
        'summary': {
            0.0: {
                'coherence': {'mean': 0.920, 'std': 0.018, 'cv': 2.8},
                'distinctiveness': {'mean': 0.720, 'std': 0.025, 'cv': 3.5},
                'diversity': {'mean': 0.620, 'std': 0.042, 'cv': 6.8},
                'semantic_integration': {'mean': 0.820, 'std': 0.021, 'cv': 2.6}
            },
            0.3: {
                'coherence': {'mean': 0.915, 'std': 0.024, 'cv': 2.6},
                'distinctiveness': {'mean': 0.715, 'std': 0.031, 'cv': 4.3},
                'diversity': {'mean': 0.625, 'std': 0.051, 'cv': 8.2},
                'semantic_integration': {'mean': 0.815, 'std': 0.028, 'cv': 3.4}
            },
            0.5: {
                'coherence': {'mean': 0.905, 'std': 0.035, 'cv': 3.9},
                'distinctiveness': {'mean': 0.705, 'std': 0.042, 'cv': 6.0},
                'diversity': {'mean': 0.630, 'std': 0.068, 'cv': 10.8},
                'semantic_integration': {'mean': 0.805, 'std': 0.038, 'cv': 4.7}
            },
            0.7: {
                'coherence': {'mean': 0.895, 'std': 0.051, 'cv': 5.7},
                'distinctiveness': {'mean': 0.695, 'std': 0.058, 'cv': 8.3},
                'diversity': {'mean': 0.635, 'std': 0.095, 'cv': 15.0},
                'semantic_integration': {'mean': 0.790, 'std': 0.054, 'cv': 6.8}
            }
        }
    }

    prompt_results = {
        'variants': [1, 2, 3, 4, 5],
        'topics': 15,
        'summary': {
            'mean_cv': 1.9,
            'classification': 'VERY LOW'
        }
    }

    report = generate_comparison_report(temp_results, prompt_results, 'files/manuscript_revision12.md')

    print(report)

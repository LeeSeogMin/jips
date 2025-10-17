"""
Robustness evaluation: Temperature and prompt sensitivity analysis.
Analyzes how sensitive LLM-based evaluations are to parameter changes.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'topic_llm'))

from evaluation.meta.report_utils import (
    bootstrap_cv, load_evaluation_results, save_evaluation_results
)


def load_parameter_variations(providers: List[str], 
                            temperatures: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
                            prompt_variants: List[str] = ['standard', 'detailed', 'concise'],
                            results_dir: str = "evaluation/outputs") -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Load evaluation results with different parameter settings.
    
    Args:
        providers: List of LLM provider names
        temperatures: List of temperature values to analyze
        prompt_variants: List of prompt variant names
        results_dir: Directory containing evaluation results
    
    Returns:
        Dictionary with structure: {provider: {dataset: {param_setting: [scores]}}}
    """
    param_results = {}
    
    for provider in providers:
        provider_results = {}
        
        for dataset in ['distinct', 'similar', 'more_similar']:
            dataset_results = {}
            
            # Load temperature variations
            for temp in temperatures:
                file_path = os.path.join(results_dir, f"{provider}_{dataset}_temp{temp}_scores.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        scores = data.get('scores', {})
                        for metric, values in scores.items():
                            key = f"temp_{temp}"
                            if key not in dataset_results:
                                dataset_results[key] = {}
                            if isinstance(values, list):
                                dataset_results[key][metric] = values
                            else:
                                dataset_results[key][metric] = [values]
                else:
                    print(f"Warning: {file_path} not found")
            
            # Load prompt variations
            for prompt_variant in prompt_variants:
                file_path = os.path.join(results_dir, f"{provider}_{dataset}_{prompt_variant}_scores.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        scores = data.get('scores', {})
                        for metric, values in scores.items():
                            key = f"prompt_{prompt_variant}"
                            if key not in dataset_results:
                                dataset_results[key] = {}
                            if isinstance(values, list):
                                dataset_results[key][metric] = values
                            else:
                                dataset_results[key][metric] = [values]
                else:
                    print(f"Warning: {file_path} not found")
            
            provider_results[dataset] = dataset_results
        
        param_results[provider] = provider_results
    
    return param_results


def compute_temperature_sensitivity(param_results: Dict[str, Dict[str, Dict[str, List[float]]]], 
                                  temperatures: List[float]) -> Dict[str, Any]:
    """
    Compute temperature sensitivity analysis.
    
    Args:
        param_results: Parameter variation results
        temperatures: List of temperature values
    
    Returns:
        Dictionary containing temperature sensitivity analysis
    """
    results = {
        'provider_sensitivity': {},
        'dataset_sensitivity': {},
        'metric_sensitivity': {}
    }
    
    # Analyze each provider
    for provider, provider_data in param_results.items():
        provider_sensitivity = {}
        
        for dataset, dataset_data in provider_data.items():
            dataset_sensitivity = {}
            
            # Get temperature variations for this dataset
            temp_scores = {}
            for temp in temperatures:
                temp_key = f"temp_{temp}"
                if temp_key in dataset_data:
                    temp_scores[temp] = dataset_data[temp_key]
            
            if len(temp_scores) >= 2:
                # Analyze each metric
                for metric in ['coherence', 'distinctiveness', 'diversity']:
                    metric_scores = {}
                    for temp, scores in temp_scores.items():
                        if metric in scores:
                            metric_scores[temp] = scores[metric]
                    
                    if len(metric_scores) >= 2:
                        # Compute sensitivity metrics
                        temp_values = list(metric_scores.keys())
                        score_values = list(metric_scores.values())
                        
                        # Coefficient of variation across temperatures
                        all_scores = []
                        for scores in score_values:
                            all_scores.extend(scores)
                        
                        if all_scores:
                            cv = bootstrap_cv(all_scores)
                            
                            # Temperature correlation
                            mean_scores = [np.mean(scores) for scores in score_values]
                            temp_corr = np.corrcoef(temp_values, mean_scores)[0, 1] if len(temp_values) > 1 else 0.0
                            
                            # Range analysis
                            score_ranges = [np.max(scores) - np.min(scores) for scores in score_values]
                            avg_range = np.mean(score_ranges)
                            max_range = np.max(score_ranges)
                            
                            dataset_sensitivity[metric] = {
                                'cv': float(cv),
                                'temp_correlation': float(temp_corr) if not np.isnan(temp_corr) else 0.0,
                                'avg_range': float(avg_range),
                                'max_range': float(max_range),
                                'n_temps': len(temp_values),
                                'temperatures': temp_values
                            }
            
            if dataset_sensitivity:
                provider_sensitivity[dataset] = dataset_sensitivity
        
        if provider_sensitivity:
            results['provider_sensitivity'][provider] = provider_sensitivity
    
    # Analyze each metric across providers and datasets
    for metric in ['coherence', 'distinctiveness', 'diversity']:
        metric_sensitivity = {}
        
        for provider, provider_data in results['provider_sensitivity'].items():
            for dataset, dataset_data in provider_data.items():
                if metric in dataset_data:
                    if provider not in metric_sensitivity:
                        metric_sensitivity[provider] = {}
                    metric_sensitivity[provider][dataset] = dataset_data[metric]
        
        if metric_sensitivity:
            results['metric_sensitivity'][metric] = metric_sensitivity
    
    return results


def compute_prompt_sensitivity(param_results: Dict[str, Dict[str, Dict[str, List[float]]]], 
                             prompt_variants: List[str]) -> Dict[str, Any]:
    """
    Compute prompt sensitivity analysis.
    
    Args:
        param_results: Parameter variation results
        prompt_variants: List of prompt variant names
    
    Returns:
        Dictionary containing prompt sensitivity analysis
    """
    results = {
        'provider_sensitivity': {},
        'dataset_sensitivity': {},
        'metric_sensitivity': {}
    }
    
    # Analyze each provider
    for provider, provider_data in param_results.items():
        provider_sensitivity = {}
        
        for dataset, dataset_data in provider_data.items():
            dataset_sensitivity = {}
            
            # Get prompt variations for this dataset
            prompt_scores = {}
            for prompt_variant in prompt_variants:
                prompt_key = f"prompt_{prompt_variant}"
                if prompt_key in dataset_data:
                    prompt_scores[prompt_variant] = dataset_data[prompt_key]
            
            if len(prompt_scores) >= 2:
                # Analyze each metric
                for metric in ['coherence', 'distinctiveness', 'diversity']:
                    metric_scores = {}
                    for prompt_variant, scores in prompt_scores.items():
                        if metric in scores:
                            metric_scores[prompt_variant] = scores[metric]
                    
                    if len(metric_scores) >= 2:
                        # Compute sensitivity metrics
                        score_values = list(metric_scores.values())
                        
                        # Coefficient of variation across prompts
                        all_scores = []
                        for scores in score_values:
                            all_scores.extend(scores)
                        
                        if all_scores:
                            cv = bootstrap_cv(all_scores)
                            
                            # ANOVA-like analysis (simplified)
                            mean_scores = [np.mean(scores) for scores in score_values]
                            std_scores = [np.std(scores) for scores in score_values]
                            
                            # Range analysis
                            score_ranges = [np.max(scores) - np.min(scores) for scores in score_values]
                            avg_range = np.mean(score_ranges)
                            max_range = np.max(score_ranges)
                            
                            # Between-prompt variance
                            overall_mean = np.mean(mean_scores)
                            between_prompt_var = np.mean([(m - overall_mean)**2 for m in mean_scores])
                            
                            dataset_sensitivity[metric] = {
                                'cv': float(cv),
                                'avg_range': float(avg_range),
                                'max_range': float(max_range),
                                'between_prompt_variance': float(between_prompt_var),
                                'n_prompts': len(score_values),
                                'prompt_variants': list(metric_scores.keys())
                            }
            
            if dataset_sensitivity:
                provider_sensitivity[dataset] = dataset_sensitivity
        
        if provider_sensitivity:
            results['provider_sensitivity'][provider] = provider_sensitivity
    
    # Analyze each metric across providers and datasets
    for metric in ['coherence', 'distinctiveness', 'diversity']:
        metric_sensitivity = {}
        
        for provider, provider_data in results['provider_sensitivity'].items():
            for dataset, dataset_data in provider_data.items():
                if metric in dataset_data:
                    if provider not in metric_sensitivity:
                        metric_sensitivity[provider] = {}
                    metric_sensitivity[provider][dataset] = dataset_data[metric]
        
        if metric_sensitivity:
            results['metric_sensitivity'][metric] = metric_sensitivity
    
    return results


def compute_parameter_stability(param_results: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Any]:
    """
    Compute overall parameter stability across all variations.
    
    Args:
        param_results: Parameter variation results
    
    Returns:
        Dictionary containing parameter stability analysis
    """
    results = {
        'provider_stability': {},
        'overall_stability': {}
    }
    
    # Analyze each provider
    for provider, provider_data in param_results.items():
        provider_stability = {}
        
        for dataset, dataset_data in provider_data.items():
            dataset_stability = {}
            
            # Collect all parameter variations
            all_variations = {}
            for param_key, param_scores in dataset_data.items():
                for metric, scores in param_scores.items():
                    if metric not in all_variations:
                        all_variations[metric] = []
                    if isinstance(scores, list):
                        all_variations[metric].append(scores)
            
            # Analyze stability for each metric
            for metric, variation_scores in all_variations.items():
                if len(variation_scores) >= 2:
                    # Compute stability metrics
                    all_scores = []
                    for scores in variation_scores:
                        all_scores.extend(scores)
                    
                    if all_scores:
                        # Overall coefficient of variation
                        cv = bootstrap_cv(all_scores)
                        
                        # Between-variation variance
                        mean_scores = [np.mean(scores) for scores in variation_scores]
                        overall_mean = np.mean(mean_scores)
                        between_var = np.mean([(m - overall_mean)**2 for m in mean_scores])
                        
                        # Within-variation variance
                        within_vars = [np.var(scores) for scores in variation_scores]
                        avg_within_var = np.mean(within_vars)
                        
                        # Stability ratio (between/within variance)
                        stability_ratio = between_var / avg_within_var if avg_within_var > 0 else 0.0
                        
                        dataset_stability[metric] = {
                            'cv': float(cv),
                            'between_variance': float(between_var),
                            'within_variance': float(avg_within_var),
                            'stability_ratio': float(stability_ratio),
                            'n_variations': len(variation_scores)
                        }
            
            if dataset_stability:
                provider_stability[dataset] = dataset_stability
        
        if provider_stability:
            results['provider_stability'][provider] = provider_stability
    
    # Compute overall stability
    all_provider_scores = {}
    for provider, provider_data in results['provider_stability'].items():
        all_scores = []
        for dataset_data in provider_data.values():
            for metric_data in dataset_data.values():
                all_scores.append(metric_data['cv'])
        all_provider_scores[provider] = all_scores
    
    if all_provider_scores:
        # Overall stability ranking
        provider_avg_cv = {}
        for provider, cvs in all_provider_scores.items():
            provider_avg_cv[provider] = np.mean(cvs)
        
        # Sort by stability (lower CV = more stable)
        sorted_providers = sorted(provider_avg_cv.items(), key=lambda x: x[1])
        
        results['overall_stability'] = {
            'provider_rankings': [{'provider': p, 'avg_cv': float(cv)} for p, cv in sorted_providers],
            'most_stable': sorted_providers[0][0] if sorted_providers else None,
            'least_stable': sorted_providers[-1][0] if sorted_providers else None
        }
    
    return results


def print_robustness_summary(temperature_results: Dict[str, Any], 
                           prompt_results: Dict[str, Any], 
                           stability_results: Dict[str, Any]) -> None:
    """Print a formatted summary of robustness results."""
    print("\n" + "="*80)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("="*80)
    
    # Temperature sensitivity
    if temperature_results['provider_sensitivity']:
        print(f"\nTemperature Sensitivity (Coefficient of Variation):")
        print("-" * 60)
        print(f"{'Provider':<12} {'Dataset':<12} {'Coherence':<12} {'Distinct':<12} {'Diversity':<12}")
        print("-" * 60)
        
        for provider, provider_data in temperature_results['provider_sensitivity'].items():
            for dataset, dataset_data in provider_data.items():
                coherence_cv = dataset_data.get('coherence', {}).get('cv', 0.0)
                distinct_cv = dataset_data.get('distinctiveness', {}).get('cv', 0.0)
                diversity_cv = dataset_data.get('diversity', {}).get('cv', 0.0)
                
                print(f"{provider:<12} {dataset:<12} {coherence_cv:<12.3f} {distinct_cv:<12.3f} {diversity_cv:<12.3f}")
    
    # Prompt sensitivity
    if prompt_results['provider_sensitivity']:
        print(f"\nPrompt Sensitivity (Coefficient of Variation):")
        print("-" * 60)
        print(f"{'Provider':<12} {'Dataset':<12} {'Coherence':<12} {'Distinct':<12} {'Diversity':<12}")
        print("-" * 60)
        
        for provider, provider_data in prompt_results['provider_sensitivity'].items():
            for dataset, dataset_data in provider_data.items():
                coherence_cv = dataset_data.get('coherence', {}).get('cv', 0.0)
                distinct_cv = dataset_data.get('distinctiveness', {}).get('cv', 0.0)
                diversity_cv = dataset_data.get('diversity', {}).get('cv', 0.0)
                
                print(f"{provider:<12} {dataset:<12} {coherence_cv:<12.3f} {distinct_cv:<12.3f} {diversity_cv:<12.3f}")
    
    # Overall stability
    if stability_results['overall_stability']:
        print(f"\nOverall Parameter Stability:")
        print("-" * 40)
        overall = stability_results['overall_stability']
        print(f"Most Stable Provider: {overall['most_stable']}")
        print(f"Least Stable Provider: {overall['least_stable']}")
        print(f"\nProvider Rankings (by stability):")
        for i, ranking in enumerate(overall['provider_rankings'], 1):
            print(f"{i}. {ranking['provider']}: CV={ranking['avg_cv']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate robustness of LLM-based topic evaluation')
    parser.add_argument('--providers', nargs='+', default=['claude', 'openai', 'grok'],
                       help='LLM providers to analyze')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help='Temperature values to analyze')
    parser.add_argument('--prompt-variants', nargs='+', default=['standard', 'detailed', 'concise'],
                       help='Prompt variant names to analyze')
    parser.add_argument('--results-dir', type=str, default='evaluation/outputs',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='evaluation/outputs/robustness_results.json',
                       help='Output file for robustness results')
    
    args = parser.parse_args()
    
    # Load parameter variations
    print(f"Loading parameter variations...")
    print(f"Temperatures: {args.temperatures}")
    print(f"Prompt variants: {args.prompt_variants}")
    
    param_results = load_parameter_variations(
        args.providers, args.temperatures, args.prompt_variants, args.results_dir
    )
    
    # Compute temperature sensitivity
    print("Computing temperature sensitivity...")
    temperature_results = compute_temperature_sensitivity(param_results, args.temperatures)
    
    # Compute prompt sensitivity
    print("Computing prompt sensitivity...")
    prompt_results = compute_prompt_sensitivity(param_results, args.prompt_variants)
    
    # Compute parameter stability
    print("Computing parameter stability...")
    stability_results = compute_parameter_stability(param_results)
    
    # Combine results
    robustness_results = {
        'temperature_sensitivity': temperature_results,
        'prompt_sensitivity': prompt_results,
        'parameter_stability': stability_results
    }
    
    # Print summary
    print_robustness_summary(temperature_results, prompt_results, stability_results)
    
    # Save results
    save_evaluation_results(robustness_results, args.output)
    print(f"\nRobustness results saved to: {args.output}")


if __name__ == '__main__':
    main()

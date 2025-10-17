"""
Reliability evaluation: Inter-rater agreement analysis using Fleiss κ and Kendall's W.
Analyzes consistency between different LLM providers and evaluation runs.
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
    fleiss_kappa, kendall_w, load_evaluation_results, save_evaluation_results
)


def load_multi_run_results(providers: List[str], n_runs: int = 3, 
                          results_dir: str = "evaluation/outputs") -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Load multiple evaluation runs for reliability analysis.
    
    Args:
        providers: List of LLM provider names
        n_runs: Number of evaluation runs to analyze
        results_dir: Directory containing evaluation results
    
    Returns:
        Dictionary with structure: {provider: {dataset: {metric: [[run1_scores], [run2_scores], ...]}}}
    """
    multi_run_results = {}
    
    for provider in providers:
        provider_results = {}
        for dataset in ['distinct', 'similar', 'more_similar']:
            dataset_results = {}
            
            for run_id in range(1, n_runs + 1):
                file_path = os.path.join(results_dir, f"{provider}_{dataset}_run{run_id}_scores.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        scores = data.get('scores', {})
                        
                        for metric, values in scores.items():
                            if metric not in dataset_results:
                                dataset_results[metric] = []
                            if isinstance(values, list):
                                dataset_results[metric].append(values)
                            else:
                                # Convert single values to lists
                                dataset_results[metric].append([values])
                else:
                    print(f"Warning: {file_path} not found")
            
            provider_results[dataset] = dataset_results
        
        multi_run_results[provider] = provider_results
    
    return multi_run_results


def compute_inter_rater_agreement(llm_scores: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Compute inter-rater agreement between different LLM providers.
    
    Args:
        llm_scores: Dictionary with structure: {provider: {dataset: {metric: [scores]}}}
    
    Returns:
        Dictionary containing inter-rater agreement analysis
    """
    results = {
        'provider_agreement': {},
        'dataset_agreement': {},
        'metric_agreement': {},
        'overall_agreement': {}
    }
    
    # Analyze each dataset
    for dataset in ['distinct', 'similar', 'more_similar']:
        dataset_agreement = {}
        
        # Get all providers' scores for this dataset
        provider_scores = {}
        for provider, provider_data in llm_scores.items():
            if dataset in provider_data and 'coherence' in provider_data[dataset]:
                provider_scores[provider] = provider_data[dataset]['coherence']
        
        if len(provider_scores) >= 2:
            # Convert to rating matrix for Fleiss κ
            all_scores = []
            for scores in provider_scores.values():
                all_scores.extend(scores)
            
            if all_scores:
                # Discretize scores into categories (e.g., 5-point scale)
                min_score, max_score = min(all_scores), max(all_scores)
                n_categories = 5
                category_width = (max_score - min_score) / n_categories
                
                # Create rating matrix
                n_subjects = len(list(provider_scores.values())[0])
                n_raters = len(provider_scores)
                ratings = np.zeros((n_subjects, n_raters), dtype=int)
                
                for i, (provider, scores) in enumerate(provider_scores.items()):
                    for j, score in enumerate(scores):
                        category = min(int((score - min_score) / category_width), n_categories - 1)
                        ratings[j, i] = category
                
                # Compute agreement metrics
                fleiss_k = fleiss_kappa(ratings)
                kendall_w_val = kendall_w(ratings)
                
                dataset_agreement = {
                    'fleiss_kappa': fleiss_k,
                    'kendall_w': kendall_w_val,
                    'n_raters': n_raters,
                    'n_subjects': n_subjects,
                    'providers': list(provider_scores.keys())
                }
                
                results['dataset_agreement'][dataset] = dataset_agreement
    
    # Analyze each metric across providers
    for metric in ['coherence', 'distinctiveness', 'diversity']:
        metric_agreement = {}
        
        for dataset in ['distinct', 'similar', 'more_similar']:
            provider_scores = {}
            for provider, provider_data in llm_scores.items():
                if (dataset in provider_data and 
                    metric in provider_data[dataset] and 
                    isinstance(provider_data[dataset][metric], list)):
                    provider_scores[provider] = provider_data[dataset][metric]
            
            if len(provider_scores) >= 2:
                # Convert to rating matrix
                all_scores = []
                for scores in provider_scores.values():
                    all_scores.extend(scores)
                
                if all_scores:
                    min_score, max_score = min(all_scores), max(all_scores)
                    n_categories = 5
                    category_width = (max_score - min_score) / n_categories
                    
                    n_subjects = len(list(provider_scores.values())[0])
                    n_raters = len(provider_scores)
                    ratings = np.zeros((n_subjects, n_raters), dtype=int)
                    
                    for i, (provider, scores) in enumerate(provider_scores.items()):
                        for j, score in enumerate(scores):
                            category = min(int((score - min_score) / category_width), n_categories - 1)
                            ratings[j, i] = category
                    
                    fleiss_k = fleiss_kappa(ratings)
                    kendall_w_val = kendall_w(ratings)
                    
                    if dataset not in metric_agreement:
                        metric_agreement[dataset] = {}
                    
                    metric_agreement[dataset] = {
                        'fleiss_kappa': fleiss_k,
                        'kendall_w': kendall_w_val,
                        'n_raters': n_raters,
                        'n_subjects': n_subjects
                    }
        
        if metric_agreement:
            results['metric_agreement'][metric] = metric_agreement
    
    # Compute overall agreement across all datasets and metrics
    all_provider_scores = {}
    for provider, provider_data in llm_scores.items():
        all_scores = []
        for dataset_data in provider_data.values():
            for metric_scores in dataset_data.values():
                if isinstance(metric_scores, list):
                    all_scores.extend(metric_scores)
        all_provider_scores[provider] = all_scores
    
    if len(all_provider_scores) >= 2:
        # Ensure all providers have the same number of scores
        min_length = min(len(scores) for scores in all_provider_scores.values())
        for provider in all_provider_scores:
            all_provider_scores[provider] = all_provider_scores[provider][:min_length]
        
        # Convert to rating matrix
        all_scores = []
        for scores in all_provider_scores.values():
            all_scores.extend(scores)
        
        if all_scores:
            min_score, max_score = min(all_scores), max(all_scores)
            n_categories = 5
            category_width = (max_score - min_score) / n_categories
            
            n_subjects = min_length
            n_raters = len(all_provider_scores)
            ratings = np.zeros((n_subjects, n_raters), dtype=int)
            
            for i, (provider, scores) in enumerate(all_provider_scores.items()):
                for j, score in enumerate(scores):
                    category = min(int((score - min_score) / category_width), n_categories - 1)
                    ratings[j, i] = category
            
            fleiss_k = fleiss_kappa(ratings)
            kendall_w_val = kendall_w(ratings)
            
            results['overall_agreement'] = {
                'fleiss_kappa': fleiss_k,
                'kendall_w': kendall_w_val,
                'n_raters': n_raters,
                'n_subjects': n_subjects,
                'providers': list(all_provider_scores.keys())
            }
    
    return results


def compute_test_retest_reliability(multi_run_results: Dict[str, Dict[str, List[List[float]]]]) -> Dict[str, Any]:
    """
    Compute test-retest reliability across multiple evaluation runs.
    
    Args:
        multi_run_results: Dictionary with multiple runs per provider
    
    Returns:
        Dictionary containing test-retest reliability analysis
    """
    results = {
        'provider_reliability': {},
        'dataset_reliability': {},
        'metric_reliability': {}
    }
    
    # Analyze each provider
    for provider, provider_data in multi_run_results.items():
        provider_reliability = {}
        
        for dataset, dataset_data in provider_data.items():
            dataset_reliability = {}
            
            for metric, runs in dataset_data.items():
                if len(runs) >= 2:
                    # Compute correlations between runs
                    correlations = []
                    for i in range(len(runs)):
                        for j in range(i + 1, len(runs)):
                            if len(runs[i]) == len(runs[j]):
                                corr = np.corrcoef(runs[i], runs[j])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        std_correlation = np.std(correlations)
                        
                        dataset_reliability[metric] = {
                            'avg_correlation': float(avg_correlation),
                            'std_correlation': float(std_correlation),
                            'n_comparisons': len(correlations),
                            'n_runs': len(runs)
                        }
            
            if dataset_reliability:
                provider_reliability[dataset] = dataset_reliability
        
        if provider_reliability:
            results['provider_reliability'][provider] = provider_reliability
    
    # Analyze each metric across providers
    for metric in ['coherence', 'distinctiveness', 'diversity']:
        metric_reliability = {}
        
        for provider, provider_data in multi_run_results.items():
            for dataset, dataset_data in provider_data.items():
                if metric in dataset_data and len(dataset_data[metric]) >= 2:
                    runs = dataset_data[metric]
                    
                    # Compute correlations between runs
                    correlations = []
                    for i in range(len(runs)):
                        for j in range(i + 1, len(runs)):
                            if len(runs[i]) == len(runs[j]):
                                corr = np.corrcoef(runs[i], runs[j])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        std_correlation = np.std(correlations)
                        
                        if provider not in metric_reliability:
                            metric_reliability[provider] = {}
                        
                        metric_reliability[provider][dataset] = {
                            'avg_correlation': float(avg_correlation),
                            'std_correlation': float(std_correlation),
                            'n_comparisons': len(correlations),
                            'n_runs': len(runs)
                        }
        
        if metric_reliability:
            results['metric_reliability'][metric] = metric_reliability
    
    return results


def print_reliability_summary(reliability_results: Dict[str, Any], 
                            test_retest_results: Dict[str, Any] = None) -> None:
    """Print a formatted summary of reliability results."""
    print("\n" + "="*80)
    print("RELIABILITY EVALUATION SUMMARY")
    print("="*80)
    
    # Inter-rater agreement
    if 'overall_agreement' in reliability_results and reliability_results['overall_agreement']:
        print(f"\nOverall Inter-Rater Agreement:")
        print("-" * 40)
        overall = reliability_results['overall_agreement']
        print(f"Fleiss κ: {overall['fleiss_kappa']:.3f}")
        print(f"Kendall's W: {overall['kendall_w']:.3f}")
        print(f"Raters: {overall['n_raters']}, Subjects: {overall['n_subjects']}")
        print(f"Providers: {', '.join(overall['providers'])}")
    
    # Dataset-wise agreement
    if reliability_results['dataset_agreement']:
        print(f"\nDataset-wise Inter-Rater Agreement:")
        print("-" * 50)
        print(f"{'Dataset':<15} {'Fleiss κ':<10} {'Kendall W':<10} {'Raters':<8} {'Subjects':<8}")
        print("-" * 50)
        
        for dataset, agreement in reliability_results['dataset_agreement'].items():
            print(f"{dataset:<15} {agreement['fleiss_kappa']:<10.3f} {agreement['kendall_w']:<10.3f} "
                  f"{agreement['n_raters']:<8} {agreement['n_subjects']:<8}")
    
    # Test-retest reliability
    if test_retest_results and test_retest_results['provider_reliability']:
        print(f"\nTest-Retest Reliability (Average Correlation):")
        print("-" * 50)
        print(f"{'Provider':<12} {'Dataset':<12} {'Coherence':<12} {'Distinct':<12} {'Diversity':<12}")
        print("-" * 50)
        
        for provider, provider_data in test_retest_results['provider_reliability'].items():
            for dataset, dataset_data in provider_data.items():
                coherence_corr = dataset_data.get('coherence', {}).get('avg_correlation', 0.0)
                distinct_corr = dataset_data.get('distinctiveness', {}).get('avg_correlation', 0.0)
                diversity_corr = dataset_data.get('diversity', {}).get('avg_correlation', 0.0)
                
                print(f"{provider:<12} {dataset:<12} {coherence_corr:<12.3f} {distinct_corr:<12.3f} {diversity_corr:<12.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate reliability of LLM-based topic evaluation')
    parser.add_argument('--providers', nargs='+', default=['claude', 'openai', 'grok'],
                       help='LLM providers to analyze')
    parser.add_argument('--n-runs', type=int, default=3,
                       help='Number of evaluation runs for test-retest analysis')
    parser.add_argument('--results-dir', type=str, default='evaluation/outputs',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='evaluation/outputs/reliability_results.json',
                       help='Output file for reliability results')
    
    args = parser.parse_args()
    
    # Load LLM scores
    print(f"Loading LLM scores for providers: {args.providers}")
    llm_scores = {}
    
    for provider in args.providers:
        provider_scores = {}
        for dataset in ['distinct', 'similar', 'more_similar']:
            file_path = os.path.join(args.results_dir, f"{provider}_{dataset}_scores.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    provider_scores[dataset] = data.get('scores', {})
            else:
                print(f"Warning: {file_path} not found")
                provider_scores[dataset] = {}
        
        llm_scores[provider] = provider_scores
    
    # Compute inter-rater agreement
    print("Computing inter-rater agreement...")
    reliability_results = compute_inter_rater_agreement(llm_scores)
    
    # Load multi-run results for test-retest analysis
    print(f"Loading multi-run results (n_runs={args.n_runs})...")
    multi_run_results = load_multi_run_results(args.providers, args.n_runs, args.results_dir)
    
    # Compute test-retest reliability
    test_retest_results = None
    if any(any(any(len(runs) >= 2 for runs in dataset_data.values()) 
               for dataset_data in provider_data.values()) 
           for provider_data in multi_run_results.values()):
        print("Computing test-retest reliability...")
        test_retest_results = compute_test_retest_reliability(multi_run_results)
        reliability_results['test_retest'] = test_retest_results
    
    # Print summary
    print_reliability_summary(reliability_results, test_retest_results)
    
    # Save results
    save_evaluation_results(reliability_results, args.output)
    print(f"\nReliability results saved to: {args.output}")


if __name__ == '__main__':
    main()

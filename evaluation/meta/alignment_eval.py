"""
Alignment evaluation: Compare DL/ST metrics against LLM provider-wise alignment.
Analyzes how well statistical and semantic metrics align with LLM judgments.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'topic_llm'))

from evaluation.meta.report_utils import (
    spearman_correlation, kendall_correlation, pairwise_accuracy,
    load_evaluation_results, save_evaluation_results, format_correlation_matrix
)


def load_llm_scores(providers: List[str], results_dir: str = "evaluation/outputs") -> Dict[str, Dict[str, List[float]]]:
    """
    Load LLM evaluation scores for multiple providers.
    
    Args:
        providers: List of LLM provider names (e.g., ['claude', 'openai', 'grok'])
        results_dir: Directory containing LLM evaluation results
    
    Returns:
        Dictionary with structure: {provider: {dataset: {metric: [scores]}}}
    """
    llm_scores = {}
    
    for provider in providers:
        provider_scores = {}
        for dataset in ['distinct', 'similar', 'more_similar']:
            file_path = os.path.join(results_dir, f"{provider}_{dataset}_scores.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    provider_scores[dataset] = data.get('scores', {})
            else:
                print(f"Warning: {file_path} not found")
                provider_scores[dataset] = {}
        
        llm_scores[provider] = provider_scores
    
    return llm_scores


def compute_alignment_metrics(dl_scores: Dict[str, List[float]], 
                            st_scores: Dict[str, List[float]], 
                            llm_scores: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Compute alignment metrics between DL/ST and LLM scores.
    
    Args:
        dl_scores: DL evaluation scores per dataset
        st_scores: ST evaluation scores per dataset  
        llm_scores: LLM scores per provider per dataset
    
    Returns:
        Dictionary containing alignment analysis results
    """
    results = {
        'datasets': {},
        'provider_summary': {},
        'overall_correlations': {}
    }
    
    # Analyze each dataset
    for dataset in ['distinct', 'similar', 'more_similar']:
        if dataset not in dl_scores or dataset not in st_scores:
            continue
            
        dataset_results = {
            'dl_scores': dl_scores[dataset],
            'st_scores': st_scores[dataset],
            'llm_scores': {},
            'correlations': {},
            'pairwise_accuracy': {}
        }
        
        # Get LLM scores for this dataset
        for provider, provider_data in llm_scores.items():
            if dataset in provider_data and 'coherence' in provider_data[dataset]:
                llm_coherence = provider_data[dataset]['coherence']
                dataset_results['llm_scores'][provider] = llm_coherence
                
                # Compute correlations
                dl_corr = spearman_correlation(dl_scores[dataset], llm_coherence)
                st_corr = spearman_correlation(st_scores[dataset], llm_coherence)
                
                dataset_results['correlations'][provider] = {
                    'dl_spearman': dl_corr,
                    'st_spearman': st_corr,
                    'dl_kendall': kendall_correlation(dl_scores[dataset], llm_coherence),
                    'st_kendall': kendall_correlation(st_scores[dataset], llm_coherence)
                }
                
                # Compute pairwise accuracy
                dl_pw = pairwise_accuracy(llm_coherence, dl_scores[dataset])
                st_pw = pairwise_accuracy(llm_coherence, st_scores[dataset])
                
                dataset_results['pairwise_accuracy'][provider] = {
                    'dl': dl_pw,
                    'st': st_pw
                }
        
        results['datasets'][dataset] = dataset_results
    
    # Compute provider-wise summary
    for provider in llm_scores.keys():
        provider_metrics = {
            'avg_dl_spearman': 0.0,
            'avg_st_spearman': 0.0,
            'avg_dl_pairwise': 0.0,
            'avg_st_pairwise': 0.0,
            'datasets_analyzed': 0
        }
        
        valid_datasets = 0
        for dataset, dataset_data in results['datasets'].items():
            if provider in dataset_data['correlations']:
                provider_metrics['avg_dl_spearman'] += dataset_data['correlations'][provider]['dl_spearman']
                provider_metrics['avg_st_spearman'] += dataset_data['correlations'][provider]['st_spearman']
                provider_metrics['avg_dl_pairwise'] += dataset_data['pairwise_accuracy'][provider]['dl']
                provider_metrics['avg_st_pairwise'] += dataset_data['pairwise_accuracy'][provider]['st']
                valid_datasets += 1
        
        if valid_datasets > 0:
            provider_metrics['avg_dl_spearman'] /= valid_datasets
            provider_metrics['avg_st_spearman'] /= valid_datasets
            provider_metrics['avg_dl_pairwise'] /= valid_datasets
            provider_metrics['avg_st_pairwise'] /= valid_datasets
            provider_metrics['datasets_analyzed'] = valid_datasets
        
        results['provider_summary'][provider] = provider_metrics
    
    # Compute overall correlations
    all_dl_scores = []
    all_st_scores = []
    all_llm_scores = {provider: [] for provider in llm_scores.keys()}
    
    for dataset in ['distinct', 'similar', 'more_similar']:
        if dataset in dl_scores and dataset in st_scores:
            all_dl_scores.extend(dl_scores[dataset])
            all_st_scores.extend(st_scores[dataset])
            
            for provider in llm_scores.keys():
                if (dataset in results['datasets'] and 
                    provider in results['datasets'][dataset]['llm_scores']):
                    all_llm_scores[provider].extend(results['datasets'][dataset]['llm_scores'][provider])
    
    for provider, scores in all_llm_scores.items():
        if len(scores) > 0 and len(scores) == len(all_dl_scores):
            results['overall_correlations'][provider] = {
                'dl_spearman': spearman_correlation(all_dl_scores, scores),
                'st_spearman': spearman_correlation(all_st_scores, scores),
                'dl_kendall': kendall_correlation(all_dl_scores, scores),
                'st_kendall': kendall_correlation(all_st_scores, scores)
            }
    
    return results


def print_alignment_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of alignment results."""
    print("\n" + "="*80)
    print("ALIGNMENT EVALUATION SUMMARY")
    print("="*80)
    
    # Provider-wise summary
    print("\nProvider-wise Alignment (Average across datasets):")
    print("-" * 60)
    print(f"{'Provider':<12} {'DL Spearman':<12} {'ST Spearman':<12} {'DL Pairwise':<12} {'ST Pairwise':<12}")
    print("-" * 60)
    
    for provider, metrics in results['provider_summary'].items():
        print(f"{provider:<12} {metrics['avg_dl_spearman']:<12.3f} {metrics['avg_st_spearman']:<12.3f} "
              f"{metrics['avg_dl_pairwise']:<12.3f} {metrics['avg_st_pairwise']:<12.3f}")
    
    # Overall correlations
    if results['overall_correlations']:
        print(f"\nOverall Correlations (All datasets combined):")
        print("-" * 60)
        print(f"{'Provider':<12} {'DL Spearman':<12} {'ST Spearman':<12} {'DL Kendall':<12} {'ST Kendall':<12}")
        print("-" * 60)
        
        for provider, corrs in results['overall_correlations'].items():
            print(f"{provider:<12} {corrs['dl_spearman']:<12.3f} {corrs['st_spearman']:<12.3f} "
                  f"{corrs['dl_kendall']:<12.3f} {corrs['st_kendall']:<12.3f}")
    
    # Dataset-wise details
    print(f"\nDataset-wise Details:")
    print("-" * 60)
    for dataset, data in results['datasets'].items():
        print(f"\n{dataset.upper()} Dataset:")
        if data['correlations']:
            print(f"{'Provider':<12} {'DL Spearman':<12} {'ST Spearman':<12} {'DL Pairwise':<12} {'ST Pairwise':<12}")
            print("-" * 60)
            for provider in data['correlations'].keys():
                corrs = data['correlations'][provider]
                pw_acc = data['pairwise_accuracy'][provider]
                print(f"{provider:<12} {corrs['dl_spearman']:<12.3f} {corrs['st_spearman']:<12.3f} "
                      f"{pw_acc['dl']:<12.3f} {pw_acc['st']:<12.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate alignment between DL/ST and LLM metrics')
    parser.add_argument('--dl-results', type=str, default='evaluation/outputs/dl_results.json',
                       help='Path to DL evaluation results JSON file')
    parser.add_argument('--st-results', type=str, default='evaluation/outputs/st_results.json', 
                       help='Path to ST evaluation results JSON file')
    parser.add_argument('--providers', nargs='+', default=['claude', 'openai', 'grok'],
                       help='LLM providers to analyze')
    parser.add_argument('--output', type=str, default='evaluation/outputs/alignment_results.json',
                       help='Output file for alignment results')
    parser.add_argument('--results-dir', type=str, default='evaluation/outputs',
                       help='Directory containing LLM evaluation results')
    
    args = parser.parse_args()
    
    # Load DL and ST results
    print("Loading DL and ST evaluation results...")
    dl_results = load_evaluation_results(args.dl_results)
    st_results = load_evaluation_results(args.st_results)
    
    # Extract coherence scores
    dl_scores = {}
    st_scores = {}
    
    for dataset in ['distinct', 'similar', 'more_similar']:
        if 'datasets' in dl_results and dataset in dl_results['datasets']:
            dl_scores[dataset] = dl_results['datasets'][dataset].get('coherence_scores', [])
        if 'datasets' in st_results and dataset in st_results['datasets']:
            st_scores[dataset] = st_results['datasets'][dataset].get('coherence_scores', [])
    
    # Load LLM scores
    print(f"Loading LLM scores for providers: {args.providers}")
    llm_scores = load_llm_scores(args.providers, args.results_dir)
    
    # Compute alignment metrics
    print("Computing alignment metrics...")
    alignment_results = compute_alignment_metrics(dl_scores, st_scores, llm_scores)
    
    # Print summary
    print_alignment_summary(alignment_results)
    
    # Save results
    save_evaluation_results(alignment_results, args.output)
    print(f"\nAlignment results saved to: {args.output}")


if __name__ == '__main__':
    main()

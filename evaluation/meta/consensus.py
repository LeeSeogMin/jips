"""
Consensus evaluation: Multi-LLM consensus methods for robust evaluation.
Implements various consensus strategies to combine multiple LLM evaluations.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'topic_llm'))

from evaluation.meta.report_utils import (
    load_evaluation_results, save_evaluation_results
)


def load_multi_llm_scores(providers: List[str], 
                         results_dir: str = "evaluation/outputs") -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Load evaluation scores from multiple LLM providers.
    
    Args:
        providers: List of LLM provider names
        results_dir: Directory containing evaluation results
    
    Returns:
        Dictionary with structure: {provider: {dataset: {metric: [scores]}}}
    """
    multi_llm_scores = {}
    
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
        
        multi_llm_scores[provider] = provider_scores
    
    return multi_llm_scores


def compute_simple_consensus(multi_llm_scores: Dict[str, Dict[str, Dict[str, List[float]]]], 
                           method: str = 'mean') -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Compute simple consensus using mean, median, or mode.
    
    Args:
        multi_llm_scores: Multi-LLM evaluation scores
        method: Consensus method ('mean', 'median', 'mode')
    
    Returns:
        Dictionary containing consensus scores
    """
    consensus_scores = {}
    
    # Get all datasets
    all_datasets = set()
    for provider_data in multi_llm_scores.values():
        all_datasets.update(provider_data.keys())
    
    for dataset in all_datasets:
        dataset_consensus = {}
        
        # Get all metrics
        all_metrics = set()
        for provider_data in multi_llm_scores.values():
            if dataset in provider_data:
                all_metrics.update(provider_data[dataset].keys())
        
        for metric in all_metrics:
            # Collect scores from all providers for this metric
            metric_scores = []
            for provider, provider_data in multi_llm_scores.items():
                if (dataset in provider_data and 
                    metric in provider_data[dataset] and 
                    isinstance(provider_data[dataset][metric], list)):
                    metric_scores.append(provider_data[dataset][metric])
            
            if metric_scores:
                # Ensure all score lists have the same length
                min_length = min(len(scores) for scores in metric_scores)
                aligned_scores = [scores[:min_length] for scores in metric_scores]
                
                # Compute consensus
                consensus_values = []
                for i in range(min_length):
                    values = [scores[i] for scores in aligned_scores]
                    
                    if method == 'mean':
                        consensus_values.append(np.mean(values))
                    elif method == 'median':
                        consensus_values.append(np.median(values))
                    elif method == 'mode':
                        # Use most frequent value (or mean if tie)
                        unique, counts = np.unique(values, return_counts=True)
                        mode_idx = np.argmax(counts)
                        if counts[mode_idx] > 1:
                            consensus_values.append(unique[mode_idx])
                        else:
                            consensus_values.append(np.mean(values))
                    else:
                        consensus_values.append(np.mean(values))
                
                dataset_consensus[metric] = consensus_values
        
        if dataset_consensus:
            consensus_scores[dataset] = dataset_consensus
    
    return consensus_scores


def compute_weighted_consensus(multi_llm_scores: Dict[str, Dict[str, Dict[str, List[float]]]], 
                              weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Compute weighted consensus based on provider reliability.
    
    Args:
        multi_llm_scores: Multi-LLM evaluation scores
        weights: Optional weights for each provider (if None, compute from data)
    
    Returns:
        Dictionary containing weighted consensus scores
    """
    consensus_scores = {}
    
    # Compute weights if not provided
    if weights is None:
        weights = compute_provider_weights(multi_llm_scores)
    
    # Get all datasets
    all_datasets = set()
    for provider_data in multi_llm_scores.values():
        all_datasets.update(provider_data.keys())
    
    for dataset in all_datasets:
        dataset_consensus = {}
        
        # Get all metrics
        all_metrics = set()
        for provider_data in multi_llm_scores.values():
            if dataset in provider_data:
                all_metrics.update(provider_data[dataset].keys())
        
        for metric in all_metrics:
            # Collect scores and weights
            metric_scores = []
            metric_weights = []
            
            for provider, provider_data in multi_llm_scores.items():
                if (dataset in provider_data and 
                    metric in provider_data[dataset] and 
                    isinstance(provider_data[dataset][metric], list)):
                    metric_scores.append(provider_data[dataset][metric])
                    metric_weights.append(weights.get(provider, 1.0))
            
            if metric_scores:
                # Ensure all score lists have the same length
                min_length = min(len(scores) for scores in metric_scores)
                aligned_scores = [scores[:min_length] for scores in metric_scores]
                
                # Normalize weights
                total_weight = sum(metric_weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in metric_weights]
                else:
                    normalized_weights = [1.0 / len(metric_weights)] * len(metric_weights)
                
                # Compute weighted consensus
                consensus_values = []
                for i in range(min_length):
                    values = [scores[i] for scores in aligned_scores]
                    weighted_sum = sum(v * w for v, w in zip(values, normalized_weights))
                    consensus_values.append(weighted_sum)
                
                dataset_consensus[metric] = consensus_values
        
        if dataset_consensus:
            consensus_scores[dataset] = dataset_consensus
    
    return consensus_scores


def compute_provider_weights(multi_llm_scores: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, float]:
    """
    Compute provider weights based on consistency and agreement.
    
    Args:
        multi_llm_scores: Multi-LLM evaluation scores
    
    Returns:
        Dictionary containing provider weights
    """
    weights = {}
    
    # Compute consistency for each provider
    for provider, provider_data in multi_llm_scores.items():
        consistency_scores = []
        
        for dataset, dataset_data in provider_data.items():
            for metric, scores in dataset_data.items():
                if isinstance(scores, list) and len(scores) > 1:
                    # Compute coefficient of variation (lower = more consistent)
                    cv = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else 0.0
                    consistency_scores.append(1.0 / (1.0 + cv))  # Convert to consistency score
        
        if consistency_scores:
            weights[provider] = np.mean(consistency_scores)
        else:
            weights[provider] = 1.0
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {provider: weight / total_weight for provider, weight in weights.items()}
    
    return weights


def compute_robust_consensus(multi_llm_scores: Dict[str, Dict[str, Dict[str, List[float]]]], 
                            outlier_threshold: float = 2.0) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Compute robust consensus by removing outliers.
    
    Args:
        multi_llm_scores: Multi-LLM evaluation scores
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Dictionary containing robust consensus scores
    """
    consensus_scores = {}
    
    # Get all datasets
    all_datasets = set()
    for provider_data in multi_llm_scores.values():
        all_datasets.update(provider_data.keys())
    
    for dataset in all_datasets:
        dataset_consensus = {}
        
        # Get all metrics
        all_metrics = set()
        for provider_data in multi_llm_scores.values():
            if dataset in provider_data:
                all_metrics.update(provider_data[dataset].keys())
        
        for metric in all_metrics:
            # Collect scores from all providers for this metric
            metric_scores = []
            for provider, provider_data in multi_llm_scores.items():
                if (dataset in provider_data and 
                    metric in provider_data[dataset] and 
                    isinstance(provider_data[dataset][metric], list)):
                    metric_scores.append(provider_data[dataset][metric])
            
            if metric_scores:
                # Ensure all score lists have the same length
                min_length = min(len(scores) for scores in metric_scores)
                aligned_scores = [scores[:min_length] for scores in metric_scores]
                
                # Compute robust consensus
                consensus_values = []
                for i in range(min_length):
                    values = [scores[i] for scores in aligned_scores]
                    
                    # Remove outliers using Z-score
                    if len(values) > 2:
                        z_scores = np.abs(stats.zscore(values))
                        filtered_values = [v for v, z in zip(values, z_scores) if z <= outlier_threshold]
                        
                        if len(filtered_values) > 0:
                            consensus_values.append(np.mean(filtered_values))
                        else:
                            consensus_values.append(np.mean(values))
                    else:
                        consensus_values.append(np.mean(values))
                
                dataset_consensus[metric] = consensus_values
        
        if dataset_consensus:
            consensus_scores[dataset] = dataset_consensus
    
    return consensus_scores


def compute_consensus_quality(consensus_scores: Dict[str, Dict[str, Dict[str, List[float]]]], 
                            multi_llm_scores: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Any]:
    """
    Compute quality metrics for consensus scores.
    
    Args:
        consensus_scores: Consensus evaluation scores
        multi_llm_scores: Original multi-LLM scores
    
    Returns:
        Dictionary containing consensus quality metrics
    """
    quality_metrics = {
        'dataset_quality': {},
        'overall_quality': {}
    }
    
    # Analyze each dataset
    for dataset, dataset_consensus in consensus_scores.items():
        dataset_quality = {}
        
        for metric, consensus_values in dataset_consensus.items():
            # Collect original scores for this metric
            original_scores = []
            for provider, provider_data in multi_llm_scores.items():
                if (dataset in provider_data and 
                    metric in provider_data[dataset] and 
                    isinstance(provider_data[dataset][metric], list)):
                    original_scores.append(provider_data[dataset][metric])
            
            if original_scores:
                # Ensure all score lists have the same length
                min_length = min(len(scores) for scores in original_scores)
                aligned_scores = [scores[:min_length] for scores in original_scores]
                consensus_aligned = consensus_values[:min_length]
                
                # Compute quality metrics
                correlations = []
                for scores in aligned_scores:
                    if len(scores) > 1:
                        corr = np.corrcoef(scores, consensus_aligned)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                avg_correlation = np.mean(correlations) if correlations else 0.0
                std_correlation = np.std(correlations) if correlations else 0.0
                
                # Compute consensus stability (lower variance = more stable)
                consensus_stability = 1.0 / (1.0 + np.var(consensus_aligned)) if len(consensus_aligned) > 1 else 1.0
                
                dataset_quality[metric] = {
                    'avg_correlation': float(avg_correlation),
                    'std_correlation': float(std_correlation),
                    'consensus_stability': float(consensus_stability),
                    'n_providers': len(original_scores)
                }
        
        if dataset_quality:
            quality_metrics['dataset_quality'][dataset] = dataset_quality
    
    # Compute overall quality
    all_correlations = []
    all_stabilities = []
    
    for dataset_data in quality_metrics['dataset_quality'].values():
        for metric_data in dataset_data.values():
            all_correlations.append(metric_data['avg_correlation'])
            all_stabilities.append(metric_data['consensus_stability'])
    
    if all_correlations:
        quality_metrics['overall_quality'] = {
            'avg_correlation': float(np.mean(all_correlations)),
            'std_correlation': float(np.std(all_correlations)),
            'avg_stability': float(np.mean(all_stabilities)),
            'std_stability': float(np.std(all_stabilities))
        }
    
    return quality_metrics


def print_consensus_summary(consensus_scores: Dict[str, Dict[str, Dict[str, List[float]]]], 
                          quality_metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of consensus results."""
    print("\n" + "="*80)
    print("CONSENSUS EVALUATION SUMMARY")
    print("="*80)
    
    # Dataset-wise consensus scores
    print(f"\nConsensus Scores by Dataset:")
    print("-" * 60)
    for dataset, dataset_scores in consensus_scores.items():
        print(f"\n{dataset.upper()} Dataset:")
        for metric, scores in dataset_scores.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  {metric}: {mean_score:.3f} ± {std_score:.3f} (n={len(scores)})")
    
    # Quality metrics
    if quality_metrics['overall_quality']:
        print(f"\nConsensus Quality Metrics:")
        print("-" * 40)
        overall = quality_metrics['overall_quality']
        print(f"Average Correlation: {overall['avg_correlation']:.3f} ± {overall['std_correlation']:.3f}")
        print(f"Average Stability: {overall['avg_stability']:.3f} ± {overall['std_stability']:.3f}")
    
    # Dataset-wise quality
    if quality_metrics['dataset_quality']:
        print(f"\nDataset-wise Quality:")
        print("-" * 50)
        print(f"{'Dataset':<12} {'Metric':<12} {'Correlation':<12} {'Stability':<12}")
        print("-" * 50)
        
        for dataset, dataset_quality in quality_metrics['dataset_quality'].items():
            for metric, metrics in dataset_quality.items():
                print(f"{dataset:<12} {metric:<12} {metrics['avg_correlation']:<12.3f} {metrics['consensus_stability']:<12.3f}")


def main():
    parser = argparse.ArgumentParser(description='Compute multi-LLM consensus for topic evaluation')
    parser.add_argument('--providers', nargs='+', default=['claude', 'openai', 'grok'],
                       help='LLM providers to include in consensus')
    parser.add_argument('--method', choices=['simple', 'weighted', 'robust'], default='weighted',
                       help='Consensus method to use')
    parser.add_argument('--simple-method', choices=['mean', 'median', 'mode'], default='mean',
                       help='Method for simple consensus')
    parser.add_argument('--outlier-threshold', type=float, default=2.0,
                       help='Z-score threshold for outlier detection in robust consensus')
    parser.add_argument('--results-dir', type=str, default='evaluation/outputs',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='evaluation/outputs/consensus_results.json',
                       help='Output file for consensus results')
    
    args = parser.parse_args()
    
    # Load multi-LLM scores
    print(f"Loading multi-LLM scores for providers: {args.providers}")
    multi_llm_scores = load_multi_llm_scores(args.providers, args.results_dir)
    
    # Compute consensus
    print(f"Computing consensus using method: {args.method}")
    
    if args.method == 'simple':
        consensus_scores = compute_simple_consensus(multi_llm_scores, args.simple_method)
    elif args.method == 'weighted':
        consensus_scores = compute_weighted_consensus(multi_llm_scores)
    elif args.method == 'robust':
        consensus_scores = compute_robust_consensus(multi_llm_scores, args.outlier_threshold)
    else:
        raise ValueError(f"Unknown consensus method: {args.method}")
    
    # Compute quality metrics
    print("Computing consensus quality metrics...")
    quality_metrics = compute_consensus_quality(consensus_scores, multi_llm_scores)
    
    # Print summary
    print_consensus_summary(consensus_scores, quality_metrics)
    
    # Save results
    results = {
        'consensus_scores': consensus_scores,
        'quality_metrics': quality_metrics,
        'method': args.method,
        'providers': args.providers
    }
    
    save_evaluation_results(results, args.output)
    print(f"\nConsensus results saved to: {args.output}")


if __name__ == '__main__':
    main()

"""
파라미터 비교 분석: 기존 vs 최적화된 파라미터

기존 파라미터:
- α=0.4, β=0.4, γ=0.2, λ=0.2
- PageRank damping=0.85 (기본값)
- Similarity threshold=0.3

최적화된 파라미터:
- α=0.6, β=0.5, γ=0.3, λ=0.2
- PageRank damping=0.75
- Similarity threshold=0.1
"""

import sys
import os
sys.path.append('evaluation')

import numpy as np
import pandas as pd
import json
from NeuralEvaluator import TopicModelNeuralEvaluator

def run_parameter_comparison():
    """파라미터 비교 분석 실행"""
    
    print("=" * 80)
    print("PARAMETER COMPARISON ANALYSIS")
    print("=" * 80)
    
    # 평가기 초기화
    evaluator = TopicModelNeuralEvaluator()
    
    # 데이터셋별 평가
    datasets = ['distinct', 'similar', 'more_similar']
    results = {}
    
    for dataset in datasets:
        print(f"\nEvaluating {dataset} topics...")
        
        try:
            # 평가 실행
            evaluation_result = evaluator.evaluate_topics(dataset)
            
            # 결과 저장
            results[dataset] = {
                'coherence': evaluation_result.get('coherence', 0),
                'distinctiveness': evaluation_result.get('distinctiveness', 0),
                'diversity': evaluation_result.get('diversity', 0),
                'integration': evaluation_result.get('integration', 0),
                'overall_score': evaluation_result.get('overall_score', 0)
            }
            
            print(f"  Coherence: {results[dataset]['coherence']:.4f}")
            print(f"  Distinctiveness: {results[dataset]['distinctiveness']:.4f}")
            print(f"  Diversity: {results[dataset]['diversity']:.4f}")
            print(f"  Integration: {results[dataset]['integration']:.4f}")
            print(f"  Overall Score: {results[dataset]['overall_score']:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {dataset}: {e}")
            results[dataset] = {
                'coherence': 0, 'distinctiveness': 0, 'diversity': 0,
                'integration': 0, 'overall_score': 0
            }
    
    # 기존 결과 로드 (Table 3 비교용)
    original_results = load_original_results()
    
    # 비교 분석
    comparison = analyze_parameter_impact(results, original_results)
    
    # 결과 저장
    save_comparison_results(results, original_results, comparison)
    
    return results, original_results, comparison

def load_original_results():
    """기존 결과 로드 (Table 3 비교용)"""
    
    # 기존 evaluation 결과에서 로드
    original_results = {
        'distinct': {
            'coherence': 0.6346,  # ST_Eval 결과
            'distinctiveness': 0.2030,
            'diversity': 0.7733,
            'overall_score': 0.5329
        },
        'similar': {
            'coherence': 0.5857,
            'distinctiveness': 0.1684,
            'diversity': 0.6267,
            'overall_score': 0.4687
        },
        'more_similar': {
            'coherence': 0.5850,
            'distinctiveness': 0.2121,
            'diversity': 0.6250,
            'overall_score': 0.4811
        }
    }
    
    return original_results

def analyze_parameter_impact(new_results, original_results):
    """파라미터 영향 분석"""
    
    print("\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)
    
    comparison = {}
    
    for dataset in ['distinct', 'similar', 'more_similar']:
        print(f"\n{dataset.upper()} TOPICS:")
        print("-" * 40)
        
        new = new_results[dataset]
        orig = original_results[dataset]
        
        # 변화량 계산
        coherence_change = ((new['coherence'] - orig['coherence']) / orig['coherence']) * 100
        distinctiveness_change = ((new['distinctiveness'] - orig['distinctiveness']) / orig['distinctiveness']) * 100
        diversity_change = ((new['diversity'] - orig['diversity']) / orig['diversity']) * 100
        overall_change = ((new['overall_score'] - orig['overall_score']) / orig['overall_score']) * 100
        
        comparison[dataset] = {
            'coherence_change': coherence_change,
            'distinctiveness_change': distinctiveness_change,
            'diversity_change': diversity_change,
            'overall_change': overall_change,
            'new_values': new,
            'original_values': orig
        }
        
        print(f"Coherence:      {orig['coherence']:.4f} → {new['coherence']:.4f} ({coherence_change:+.1f}%)")
        print(f"Distinctiveness: {orig['distinctiveness']:.4f} → {new['distinctiveness']:.4f} ({distinctiveness_change:+.1f}%)")
        print(f"Diversity:      {orig['diversity']:.4f} → {new['diversity']:.4f} ({diversity_change:+.1f}%)")
        print(f"Overall Score:  {orig['overall_score']:.4f} → {new['overall_score']:.4f} ({overall_change:+.1f}%)")
    
    # 전체 요약
    print(f"\n" + "=" * 80)
    print("OVERALL IMPACT SUMMARY")
    print("=" * 80)
    
    avg_coherence_change = np.mean([comparison[ds]['coherence_change'] for ds in comparison])
    avg_distinctiveness_change = np.mean([comparison[ds]['distinctiveness_change'] for ds in comparison])
    avg_diversity_change = np.mean([comparison[ds]['diversity_change'] for ds in comparison])
    avg_overall_change = np.mean([comparison[ds]['overall_change'] for ds in comparison])
    
    print(f"Average Coherence Change:      {avg_coherence_change:+.1f}%")
    print(f"Average Distinctiveness Change: {avg_distinctiveness_change:+.1f}%")
    print(f"Average Diversity Change:      {avg_diversity_change:+.1f}%")
    print(f"Average Overall Score Change:  {avg_overall_change:+.1f}%")
    
    # 파라미터별 영향 분석
    print(f"\n" + "=" * 80)
    print("PARAMETER-SPECIFIC IMPACT ANALYSIS")
    print("=" * 80)
    
    print("α (coherence weight): 0.4 → 0.6 (+50%)")
    print("  - Expected: Higher coherence scores due to increased weight")
    print(f"  - Actual: {avg_coherence_change:+.1f}% change in coherence")
    
    print("\nβ (distinctiveness weight): 0.4 → 0.5 (+25%)")
    print("  - Expected: Higher distinctiveness scores due to increased weight")
    print(f"  - Actual: {avg_distinctiveness_change:+.1f}% change in distinctiveness")
    
    print("\nγ (diversity weight): 0.2 → 0.3 (+50%)")
    print("  - Expected: Higher diversity scores due to increased weight")
    print(f"  - Actual: {avg_diversity_change:+.1f}% change in diversity")
    
    print("\nλ (integration weight): 0.2 → 0.2 (no change)")
    print("  - Expected: No change in integration scores")
    
    print("\nPageRank damping: 0.85 → 0.75 (-12%)")
    print("  - Expected: Different importance weighting in coherence calculation")
    
    print("\nSimilarity threshold: 0.3 → 0.1 (-67%)")
    print("  - Expected: More edges in graph, different PageRank results")
    
    return comparison

def save_comparison_results(new_results, original_results, comparison):
    """비교 결과 저장"""
    
    # 결과를 DataFrame으로 변환
    data = []
    for dataset in ['distinct', 'similar', 'more_similar']:
        new = new_results[dataset]
        orig = original_results[dataset]
        comp = comparison[dataset]
        
        data.append({
            'dataset': dataset,
            'metric': 'coherence',
            'original': orig['coherence'],
            'optimized': new['coherence'],
            'change_pct': comp['coherence_change']
        })
        data.append({
            'dataset': dataset,
            'metric': 'distinctiveness',
            'original': orig['distinctiveness'],
            'optimized': new['distinctiveness'],
            'change_pct': comp['distinctiveness_change']
        })
        data.append({
            'dataset': dataset,
            'metric': 'diversity',
            'original': orig['diversity'],
            'optimized': new['diversity'],
            'change_pct': comp['diversity_change']
        })
        data.append({
            'dataset': dataset,
            'metric': 'overall_score',
            'original': orig['overall_score'],
            'optimized': new['overall_score'],
            'change_pct': comp['overall_change']
        })
    
    df = pd.DataFrame(data)
    df.to_csv('grid_search_optimization/parameter_comparison_results.csv', index=False)
    
    # JSON 형태로도 저장
    comparison_data = {
        'parameter_changes': {
            'alpha': {'original': 0.4, 'optimized': 0.6, 'change_pct': 50.0},
            'beta': {'original': 0.4, 'optimized': 0.5, 'change_pct': 25.0},
            'gamma': {'original': 0.2, 'optimized': 0.3, 'change_pct': 50.0},
            'lambda': {'original': 0.2, 'optimized': 0.2, 'change_pct': 0.0},
            'damping': {'original': 0.85, 'optimized': 0.75, 'change_pct': -11.8},
            'threshold': {'original': 0.3, 'optimized': 0.1, 'change_pct': -66.7}
        },
        'performance_impact': comparison,
        'summary': {
            'avg_coherence_change': np.mean([comparison[ds]['coherence_change'] for ds in comparison]),
            'avg_distinctiveness_change': np.mean([comparison[ds]['distinctiveness_change'] for ds in comparison]),
            'avg_diversity_change': np.mean([comparison[ds]['diversity_change'] for ds in comparison]),
            'avg_overall_change': np.mean([comparison[ds]['overall_change'] for ds in comparison])
        }
    }
    
    with open('grid_search_optimization/parameter_comparison_analysis.json', 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print("Files saved:")
    print("  - grid_search_optimization/parameter_comparison_results.csv")
    print("  - grid_search_optimization/parameter_comparison_analysis.json")

if __name__ == "__main__":
    try:
        results, original_results, comparison = run_parameter_comparison()
        print(f"\nParameter comparison completed successfully!")
    except Exception as e:
        print(f"Error in parameter comparison: {e}")
        import traceback
        traceback.print_exc()

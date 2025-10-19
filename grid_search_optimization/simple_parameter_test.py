"""
간단한 파라미터 테스트
"""

import sys
sys.path.append('evaluation')

from NeuralEvaluator import TopicModelNeuralEvaluator
import json

def test_parameters():
    """파라미터 테스트"""
    
    print("=" * 60)
    print("PARAMETER TEST")
    print("=" * 60)
    
    # 평가기 초기화
    evaluator = TopicModelNeuralEvaluator()
    
    # 현재 파라미터 확인
    print(f"Current parameters:")
    print(f"  α (coherence): {evaluator.alpha}")
    print(f"  β (distinctiveness): {evaluator.beta}")
    print(f"  γ (diversity): {evaluator.gamma}")
    print(f"  λ (integration): {evaluator.lambda_w}")
    
    # 간단한 테스트 데이터로 평가
    test_topics = [
        ["machine", "learning", "algorithm", "neural", "network"],
        ["biology", "cell", "dna", "protein", "genetic"],
        ["history", "ancient", "civilization", "culture", "tradition"]
    ]
    
    print(f"\nTesting with sample topics...")
    
    try:
        # 평가 실행
        result = evaluator._evaluate_semantic_coherence(test_topics)
        
        print(f"Coherence evaluation result:")
        print(f"  Type: {type(result)}")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
            
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # 기존 결과와 비교할 수 있는 간단한 계산
    print(f"\n" + "=" * 60)
    print("PARAMETER IMPACT SIMULATION")
    print("=" * 60)
    
    # 기존 파라미터 (주석에서)
    original_alpha, original_beta, original_gamma, original_lambda = 0.4, 0.4, 0.2, 0.2
    
    # 현재 파라미터 (최적화된)
    current_alpha, current_beta, current_gamma, current_lambda = evaluator.alpha, evaluator.beta, evaluator.gamma, evaluator.lambda_w
    
    print(f"Parameter changes:")
    print(f"  α: {original_alpha} → {current_alpha} ({((current_alpha/original_alpha-1)*100):+.1f}%)")
    print(f"  β: {original_beta} → {current_beta} ({((current_beta/original_beta-1)*100):+.1f}%)")
    print(f"  γ: {original_gamma} → {current_gamma} ({((current_gamma/original_gamma-1)*100):+.1f}%)")
    print(f"  λ: {original_lambda} → {current_lambda} ({((current_lambda/original_lambda-1)*100):+.1f}%)")
    
    # 가상의 메트릭 값으로 영향 시뮬레이션
    mock_coherence, mock_distinctiveness, mock_diversity, mock_integration = 0.6, 0.4, 0.5, 0.3
    
    original_score = (original_alpha * mock_coherence + 
                     original_beta * mock_distinctiveness + 
                     original_gamma * mock_diversity + 
                     original_lambda * mock_integration)
    
    current_score = (current_alpha * mock_coherence + 
                    current_beta * mock_distinctiveness + 
                    current_gamma * mock_diversity + 
                    current_lambda * mock_integration)
    
    print(f"\nSimulated overall score impact:")
    print(f"  Original: {original_score:.4f}")
    print(f"  Optimized: {current_score:.4f}")
    print(f"  Change: {((current_score/original_score-1)*100):+.1f}%")
    
    # 결과 저장
    result_data = {
        "parameters": {
            "original": {
                "alpha": original_alpha,
                "beta": original_beta, 
                "gamma": original_gamma,
                "lambda": original_lambda
            },
            "optimized": {
                "alpha": current_alpha,
                "beta": current_beta,
                "gamma": current_gamma, 
                "lambda": current_lambda
            }
        },
        "simulated_impact": {
            "original_score": original_score,
            "optimized_score": current_score,
            "change_percent": ((current_score/original_score-1)*100)
        }
    }
    
    with open('grid_search_optimization/simple_parameter_test_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\nResults saved to: grid_search_optimization/simple_parameter_test_results.json")

if __name__ == "__main__":
    test_parameters()

"""
Cohen's Kappa Analysis

연속형 LLM 점수를 범주형 레이블로 변환하고 Cohen's κ를 계산하여 
inter-rater reliability를 측정하는 도구
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from typing import Dict, List, Tuple, Union, Optional
import logging
import os

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def continuous_to_categorical(scores: np.ndarray, thresholds: List[float], 
                             labels: List[str]) -> np.ndarray:
    """
    연속형 점수를 범주형 레이블로 변환
    
    Args:
        scores: 연속형 점수 배열 (범위: 0-1)
        thresholds: 분류 임계값 리스트 (예: [0.60, 0.80])
        labels: 각 범주의 레이블 (예: ['poor', 'acceptable', 'excellent'])
        
    Returns:
        np.ndarray: 범주형 레이블 배열
        
    예시:
        scores = [0.55, 0.75, 0.85]
        thresholds = [0.60, 0.80]
        labels = ['poor', 'acceptable', 'excellent']
        결과: ['poor', 'acceptable', 'excellent']
    """
    if len(thresholds) + 1 != len(labels):
        raise ValueError(f"Labels count ({len(labels)}) must be thresholds count ({len(thresholds)}) + 1")
        
    categories = []
    for score in scores:
        for i, threshold in enumerate(thresholds):
            if score < threshold:
                categories.append(labels[i])
                break
        else:
            categories.append(labels[-1])
    return np.array(categories)


def analyze_llm_agreement(scores_llm1: np.ndarray, scores_llm2: np.ndarray,
                         thresholds: Optional[List[float]] = None, 
                         labels: Optional[List[str]] = None,
                         weights: Optional[str] = None) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    두 LLM 평가자 간의 일치도 분석
    
    Args:
        scores_llm1: 첫 번째 LLM의 연속형 점수 (범위: 0-1)
        scores_llm2: 두 번째 LLM의 연속형 점수 (범위: 0-1)
        thresholds: 범주화 임계값 (기본값: [0.60, 0.80])
        labels: 범주 레이블 (기본값: ['poor', 'acceptable', 'excellent'])
        weights: 가중치 유형 (None, 'linear', 'quadratic')
        
    Returns:
        Dict: 분석 결과 딕셔너리
    """
    if thresholds is None:
        thresholds = [0.60, 0.80]
    if labels is None:
        labels = ['poor', 'acceptable', 'excellent']
        
    # 연속형 -> 범주형 변환
    cat_llm1 = continuous_to_categorical(scores_llm1, thresholds, labels)
    cat_llm2 = continuous_to_categorical(scores_llm2, thresholds, labels)
    
    # Cohen's kappa 계산
    kappa = cohen_kappa_score(cat_llm1, cat_llm2, weights=weights)
    
    # 혼동 행렬 계산
    cm = confusion_matrix(cat_llm1, cat_llm2, labels=labels)
    
    # 관찰 일치도와 기대 일치도 계산
    n_total = cm.sum()
    p_o = np.trace(cm) / n_total
    
    # 기대 일치도 계산 (우연에 의한 일치 확률)
    p_e = 0
    for i in range(len(labels)):
        # LLM1이 범주 i를 선택할 확률 × LLM2가 범주 i를 선택할 확률
        p_e += (cm[i, :].sum() / n_total) * (cm[:, i].sum() / n_total)
    
    # Cohen's kappa 해석 (Landis & Koch, 1977)
    def interpret_kappa(k):
        if k < 0:
            return "Poor (worse than random)"
        elif k < 0.20:
            return "Slight agreement"
        elif k < 0.40:
            return "Fair agreement"
        elif k < 0.60:
            return "Moderate agreement"
        elif k < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
    
    # 연속형 점수에 대한 추가 분석
    pearson_r = np.corrcoef(scores_llm1, scores_llm2)[0, 1]
    mad = np.mean(np.abs(scores_llm1 - scores_llm2))
    
    return {
        'categorical_analysis': {
            'kappa': kappa,
            'observed_agreement': p_o,
            'expected_agreement': p_e,
            'confusion_matrix': cm,
            'interpretation': interpret_kappa(kappa),
            'categories_llm1': cat_llm1,
            'categories_llm2': cat_llm2
        },
        'continuous_analysis': {
            'correlation': pearson_r,
            'mean_absolute_difference': mad,
            'llm1_mean': np.mean(scores_llm1),
            'llm2_mean': np.mean(scores_llm2),
            'llm1_std': np.std(scores_llm1),
            'llm2_std': np.std(scores_llm2)
        }
    }


def multi_rater_analysis(scores_dict: Dict[str, np.ndarray],
                        thresholds: Optional[List[float]] = None,
                        labels: Optional[List[str]] = None,
                        weights: Optional[str] = None) -> pd.DataFrame:
    """
    여러 LLM 평가자 간의 일치도 분석
    
    Args:
        scores_dict: LLM별 점수 딕셔너리 {'LLM1': scores1, 'LLM2': scores2, ...}
        thresholds: 범주화 임계값 (기본값: [0.60, 0.80])
        labels: 범주 레이블 (기본값: ['poor', 'acceptable', 'excellent'])
        weights: 가중치 유형 (None, 'linear', 'quadratic')
        
    Returns:
        pd.DataFrame: LLM 쌍별 kappa 값 행렬
    """
    if thresholds is None:
        thresholds = [0.60, 0.80]
    if labels is None:
        labels = ['poor', 'acceptable', 'excellent']
    
    # LLM 이름 리스트
    llm_names = list(scores_dict.keys())
    n_llms = len(llm_names)
    
    # 결과 행렬 초기화
    kappa_matrix = np.zeros((n_llms, n_llms))
    corr_matrix = np.zeros((n_llms, n_llms))
    mad_matrix = np.zeros((n_llms, n_llms))
    
    # 모든 LLM 쌍에 대해 kappa 계산
    for i in range(n_llms):
        for j in range(i, n_llms):
            if i == j:
                # 대각선 요소: 자기 자신과의 비교는 1
                kappa_matrix[i, j] = 1.0
                corr_matrix[i, j] = 1.0
                mad_matrix[i, j] = 0.0
            else:
                # LLM 쌍 비교 분석
                result = analyze_llm_agreement(
                    scores_dict[llm_names[i]], scores_dict[llm_names[j]], 
                    thresholds, labels, weights
                )
                # 결과 저장
                kappa_value = result['categorical_analysis']['kappa']
                corr_value = result['continuous_analysis']['correlation']
                mad_value = result['continuous_analysis']['mean_absolute_difference']
                
                # 행렬 대칭적으로 채우기
                kappa_matrix[i, j] = kappa_value
                kappa_matrix[j, i] = kappa_value
                
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value
                
                mad_matrix[i, j] = mad_value
                mad_matrix[j, i] = mad_value
    
    # 행렬을 DataFrame으로 변환
    kappa_df = pd.DataFrame(kappa_matrix, index=llm_names, columns=llm_names)
    corr_df = pd.DataFrame(corr_matrix, index=llm_names, columns=llm_names)
    mad_df = pd.DataFrame(mad_matrix, index=llm_names, columns=llm_names)
    
    return {
        'kappa': kappa_df,
        'correlation': corr_df,
        'mean_absolute_difference': mad_df
    }


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix",
                         output_path: Optional[str] = None):
    """
    혼동 행렬 시각화
    
    Args:
        cm: 혼동 행렬
        labels: 범주 레이블
        title: 그래프 제목
        output_path: 저장 경로 (None이면 저장하지 않음)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # 셀에 값 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('LLM 1')
    plt.xlabel('LLM 2')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def generate_kappa_report(results: Dict[str, Union[float, Dict, np.ndarray]], 
                         llm1_name: str = "LLM 1", llm2_name: str = "LLM 2") -> str:
    """
    Cohen's kappa 분석 결과를 마크다운 리포트로 변환
    
    Args:
        results: analyze_llm_agreement() 반환 결과
        llm1_name: 첫 번째 LLM 이름
        llm2_name: 두 번째 LLM 이름
        
    Returns:
        str: 마크다운 형식 리포트
    """
    cat_results = results['categorical_analysis']
    cont_results = results['continuous_analysis']
    
    cm = cat_results['confusion_matrix']
    
    # 혼동 행렬 문자열 변환
    cm_str = []
    labels = [str(i) for i in range(cm.shape[0])]  # 기본 레이블
    
    # labels 키가 있으면 사용
    if 'labels' in cat_results:
        labels = cat_results['labels']
    
    # 테이블 헤더
    cm_header = f"{'':10}"
    for label in labels:
        cm_header += f"{label:12}"
    cm_str.append(cm_header)
    
    # 테이블 내용
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<10}"
        for cell in row:
            row_str += f"{cell:12d}"
        cm_str.append(row_str)
    
    # 마크다운 리포트 생성
    report = [
        "# Cohen's Kappa Analysis Report\n",
        "## Inter-Rater Reliability\n",
        f"**Cohen's κ**: {cat_results['kappa']:.3f}",
        f"**Interpretation**: {cat_results['interpretation']}",
        
        "\n### Agreement Statistics\n",
        f"- Observed Agreement (p_o): {cat_results['observed_agreement']:.3f}",
        f"- Expected Agreement (p_e): {cat_results['expected_agreement']:.3f}",
        f"- Number of Samples: {cm.sum()}",
        
        "\n### Confusion Matrix\n",
        "```",
        *cm_str,
        "```",
        
        "\n## Continuous Score Analysis\n",
        f"- Pearson Correlation: {cont_results['correlation']:.3f}",
        f"- Mean Absolute Difference: {cont_results['mean_absolute_difference']:.3f}",
        
        "\n### Score Distributions\n",
        f"- {llm1_name}: μ={cont_results['llm1_mean']:.3f}, σ={cont_results['llm1_std']:.3f}",
        f"- {llm2_name}: μ={cont_results['llm2_mean']:.3f}, σ={cont_results['llm2_std']:.3f}",
    ]
    
    return "\n".join(report)


# 코드 실행 예시 (스크립트로 직접 실행하는 경우)
if __name__ == "__main__":
    # 예시 데이터
    np.random.seed(42)
    scores_llm1 = np.random.normal(0.85, 0.07, 45)  # 평균 0.85, 표준편차 0.07인 45개 샘플
    scores_llm2 = np.random.normal(0.83, 0.09, 45)  # 평균 0.83, 표준편차 0.09인 45개 샘플
    
    # 유효한 범위로 클리핑 (0-1 범위)
    scores_llm1 = np.clip(scores_llm1, 0, 1)
    scores_llm2 = np.clip(scores_llm2, 0, 1)
    
    # 분석 실행
    results = analyze_llm_agreement(
        scores_llm1, 
        scores_llm2, 
        thresholds=[0.60, 0.80],
        labels=['poor', 'acceptable', 'excellent']
    )
    
    # 보고서 생성
    report = generate_kappa_report(results, "Claude", "GPT-4")
    print(report)
    
    # 혼동 행렬 시각화 (선택 사항)
    plot_confusion_matrix(
        results['categorical_analysis']['confusion_matrix'],
        ['poor', 'acceptable', 'excellent'],
        "Claude vs GPT-4 Agreement"
    )
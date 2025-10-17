"""
LLM Robustness Analysis

다양한 조건 (temperature, prompt 변형, 다중 LLM)에서
LLM 평가의 강건성을 테스트하는 도구
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """
    LLM 평가 강건성 분석기
    
    다음 분석 지원:
    1. Temperature sensitivity analysis
    2. Prompt variation analysis
    3. Multi-LLM comparison
    4. Multi-iteration stability
    5. Mitigation strategies recommendation
    """
    
    def __init__(self, output_dir=None):
        """
        LLM 강건성 분석기 초기화
        
        Args:
            output_dir: 결과 저장 디렉토리 (기본값: ../data/robustness_results)
        """
        # 출력 디렉토리 설정
        if output_dir is None:
            base_dir = Path(__file__).parent.parent
            output_dir = base_dir / 'data' / 'robustness_results'
        else:
            output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # 데이터 디렉토리 설정
        self.data_dir = Path(__file__).parent.parent / 'data'
        
        # 결과 저장 딕셔너리
        self.results = {
            'temperature_sensitivity': {},
            'prompt_variations': {},
            'multi_llm': {},
            'multi_iteration': {},
            'mitigation_strategies': {}
        }
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Robustness Analyzer initialized. Session ID: {self.session_id}")
        logger.info(f"Results will be saved to: {self.output_dir}")
        
    def _load_topics(self, dataset: str) -> List[List[str]]:
        """
        토픽 데이터 로드
        
        Args:
            dataset: 데이터셋 이름 ('distinct', 'similar', 'more_similar')
            
        Returns:
            List[List[str]]: 토픽 키워드 리스트
        """
        pkl_file = self.data_dir / f'topics_{dataset}.pkl'
        if not pkl_file.exists():
            raise FileNotFoundError(f"Topic data not found: {pkl_file}")
            
        with open(pkl_file, 'rb') as f:
            topics = pickle.load(f)
        
        return topics
    
    def temperature_sensitivity_analysis(self, evaluator_class, 
                                        temperatures: List[float] = None,
                                        dataset: str = "distinct") -> Dict[str, Any]:
        """
        Temperature 민감도 분석
        
        Args:
            evaluator_class: LLM evaluator 클래스 (AnthropicEval, OpenAIEval 등)
            temperatures: 테스트할 temperature 값 리스트 (기본값: [0.0, 0.3, 0.7, 1.0])
            dataset: 사용할 데이터셋 이름 (기본값: "distinct")
            
        Returns:
            Dict: 분석 결과 딕셔너리
        """
        if temperatures is None:
            temperatures = [0.0, 0.3, 0.7, 1.0]
            
        logger.info(f"Starting temperature sensitivity analysis with {temperatures} on {dataset} dataset")
        
        # 토픽 데이터 로드
        try:
            topics = self._load_topics(dataset)
            logger.info(f"Loaded {len(topics)} topics from {dataset} dataset")
        except FileNotFoundError as e:
            logger.error(f"Error loading topics: {e}")
            raise
        
        results = {
            'dataset': dataset,
            'temperatures': temperatures,
            'scores': {},
            'statistics': {}
        }
        
        # 각 temperature에 대해 평가 수행
        for temp in temperatures:
            logger.info(f"Evaluating with temperature={temp}")
            
            try:
                # evaluator 인스턴스 생성
                evaluator = evaluator_class()
                
                # temperature 설정 (해당 메서드가 있는 경우)
                if hasattr(evaluator, 'set_temperature'):
                    evaluator.set_temperature(temp)
                    logger.info(f"Set temperature to {temp}")
                else:
                    logger.warning(f"Evaluator doesn't support set_temperature method")
                
                # 평가 수행
                evaluation_results = evaluator.evaluate_topic_set(topics, f"{dataset.title()} Topics (T={temp})")
                
                # 결과 저장
                results['scores'][temp] = {
                    'coherence': evaluation_results['scores']['coherence'],
                    'distinctiveness': evaluation_results['scores']['distinctiveness'],
                    'diversity': evaluation_results['scores']['diversity'],
                    'semantic_integration': evaluation_results['scores']['semantic_integration'],
                    'overall_score': evaluation_results['scores']['overall_score']
                }
                
                logger.info(f"Evaluation completed for temperature={temp}")
                
            except Exception as e:
                logger.error(f"Error evaluating with temperature={temp}: {e}")
                results['scores'][temp] = None
        
        # 온도별 통계 계산
        valid_scores = {t: s for t, s in results['scores'].items() if s is not None}
        
        if valid_scores:
            # 메트릭별 평균 및 표준편차
            metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
            stats_by_metric = {}
            
            for metric in metrics:
                values = [scores[metric] for scores in valid_scores.values()]
                stats_by_metric[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                    'range': max(values) - min(values)
                }
            
            # 전체 통계
            all_values = []
            for scores in valid_scores.values():
                all_values.extend([scores[m] for m in metrics])
                
            results['statistics'] = {
                'by_metric': stats_by_metric,
                'mean_cv': np.mean([s['cv'] for s in stats_by_metric.values()]),
                'mean_range': np.mean([s['range'] for s in stats_by_metric.values()]),
                'metrics_analyzed': metrics
            }
            
            # 등급 및 결론
            cv = results['statistics']['mean_cv']
            if cv < 0.05:
                sensitivity = "very low"
            elif cv < 0.10:
                sensitivity = "low"
            elif cv < 0.20:
                sensitivity = "moderate"
            else:
                sensitivity = "high"
                
            results['statistics']['sensitivity_grade'] = sensitivity
            results['statistics']['conclusion'] = f"Temperature sensitivity is {sensitivity} (CV={cv:.3f})"
            
            logger.info(f"Temperature sensitivity analysis completed: {sensitivity} sensitivity")
        else:
            logger.error("No valid evaluation results")
            
        # 전체 결과에 저장
        self.results['temperature_sensitivity'][dataset] = results
        
        # 시각화 생성 (옵션)
        try:
            self._plot_temperature_sensitivity(results, dataset)
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")
        
        return results
    
    def _plot_temperature_sensitivity(self, results: Dict[str, Any], dataset: str):
        """
        Temperature 민감도 분석 시각화
        
        Args:
            results: temperature_sensitivity_analysis 결과
            dataset: 데이터셋 이름
        """
        temperatures = results['temperatures']
        scores = results['scores']
        
        # 유효한 결과만 필터링
        valid_temps = [t for t in temperatures if scores[t] is not None]
        
        if not valid_temps:
            logger.warning("No valid data for visualization")
            return
        
        # 메트릭별로 그래프 생성
        metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
        
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            
            x = valid_temps
            y = [scores[t][metric] for t in valid_temps]
            
            plt.plot(x, y, 'o-', label=metric)
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.xlabel('Temperature')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 통계 정보 표시
            cv = results['statistics']['by_metric'][metric]['cv']
            plt.annotate(f"CV: {cv:.3f}", xy=(0.05, 0.05), xycoords='axes fraction')
        
        plt.tight_layout()
        
        # 저장
        output_path = self.output_dir / f"temperature_sensitivity_{dataset}_{self.session_id}.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Temperature sensitivity visualization saved to {output_path}")
    
    def prompt_variation_analysis(self, evaluator_class, dataset: str = "distinct",
                                num_variations: int = 5, prompt_type: str = "coherence") -> Dict[str, Any]:
        """
        프롬프트 변형 분석
        
        Args:
            evaluator_class: LLM evaluator 클래스 (AnthropicEval, OpenAIEval 등)
            dataset: 사용할 데이터셋 이름 (기본값: "distinct")
            num_variations: 생성할 프롬프트 변형 수
            prompt_type: 변형할 프롬프트 유형 ('coherence', 'distinctiveness', 'diversity')
            
        Returns:
            Dict: 분석 결과 딕셔너리
        """
        logger.info(f"Starting prompt variation analysis with {num_variations} variations on {dataset} dataset")
        
        # 토픽 데이터 로드
        try:
            topics = self._load_topics(dataset)
            logger.info(f"Loaded {len(topics)} topics from {dataset} dataset")
        except FileNotFoundError as e:
            logger.error(f"Error loading topics: {e}")
            raise
            
        # 기본 평가자
        base_evaluator = evaluator_class()
        
        # 프롬프트 변형 생성
        prompt_variations = self._generate_prompt_variations(prompt_type, num_variations)
        
        results = {
            'dataset': dataset,
            'prompt_type': prompt_type,
            'variations': prompt_variations,
            'scores': {},
            'statistics': {}
        }
        
        # 각 변형에 대해 평가 수행
        for idx, (name, prompt) in enumerate(prompt_variations.items()):
            logger.info(f"Evaluating with prompt variation {idx+1}/{len(prompt_variations)}: {name}")
            
            try:
                # evaluator 인스턴스 생성
                evaluator = evaluator_class()
                
                # 프롬프트 설정 (해당 메서드가 있는 경우)
                if hasattr(evaluator, 'set_system_prompt'):
                    evaluator.set_system_prompt(prompt)
                    logger.info(f"Set custom prompt for {name}")
                else:
                    logger.warning(f"Evaluator doesn't support set_system_prompt method")
                
                # 평가 수행
                evaluation_results = evaluator.evaluate_topic_set(topics, f"{dataset.title()} Topics ({name})")
                
                # 결과 저장
                results['scores'][name] = {
                    'coherence': evaluation_results['scores']['coherence'],
                    'distinctiveness': evaluation_results['scores']['distinctiveness'],
                    'diversity': evaluation_results['scores']['diversity'],
                    'semantic_integration': evaluation_results['scores']['semantic_integration'],
                    'overall_score': evaluation_results['scores']['overall_score']
                }
                
                logger.info(f"Evaluation completed for variation: {name}")
                
            except Exception as e:
                logger.error(f"Error evaluating with prompt variation {name}: {e}")
                results['scores'][name] = None
        
        # 변형별 통계 계산
        valid_scores = {v: s for v, s in results['scores'].items() if s is not None}
        
        if valid_scores:
            # 메트릭별 평균, 표준편차, CV, 범위
            metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
            stats_by_metric = {}
            
            for metric in metrics:
                values = [scores[metric] for scores in valid_scores.values()]
                stats_by_metric[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                    'range': max(values) - min(values)
                }
            
            # 전체 통계
            results['statistics'] = {
                'by_metric': stats_by_metric,
                'mean_cv': np.mean([s['cv'] for s in stats_by_metric.values()]),
                'mean_range': np.mean([s['range'] for s in stats_by_metric.values()]),
                'metrics_analyzed': metrics
            }
            
            # 등급 및 결론
            cv = results['statistics']['mean_cv']
            if cv < 0.03:
                sensitivity = "very low"
            elif cv < 0.07:
                sensitivity = "low"
            elif cv < 0.15:
                sensitivity = "moderate"
            else:
                sensitivity = "high"
                
            results['statistics']['sensitivity_grade'] = sensitivity
            results['statistics']['conclusion'] = f"Prompt sensitivity is {sensitivity} (CV={cv:.3f})"
            
            logger.info(f"Prompt variation analysis completed: {sensitivity} sensitivity")
        else:
            logger.error("No valid evaluation results")
            
        # 전체 결과에 저장
        self.results['prompt_variations'][dataset] = results
        
        # 시각화 생성 (옵션)
        try:
            self._plot_prompt_variations(results, dataset)
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")
        
        return results
    
    def _generate_prompt_variations(self, prompt_type: str, num_variations: int) -> Dict[str, str]:
        """
        프롬프트 변형 생성
        
        Args:
            prompt_type: 변형할 프롬프트 유형 ('coherence', 'distinctiveness', 'diversity')
            num_variations: 생성할 변형 수
            
        Returns:
            Dict[str, str]: 변형 이름과 프롬프트 텍스트 매핑
        """
        variations = {}
        
        # 기본 프롬프트
        base_prompt = "You are a text analysis expert. Rate word groups on a scale of 0 to 1."
        
        # 변형 1: 지시 강조
        if num_variations >= 1:
            variations["variation_1"] = (
                f"{base_prompt}\n\n"
                f"IMPORTANT: Provide an EXACT score between 0 and 1 for {prompt_type}.\n"
                f"Format your response as: <score>\n<explanation>"
            )
        
        # 변형 2: 학문적 톤
        if num_variations >= 2:
            variations["variation_2"] = (
                f"As a computational linguist specializing in topic model evaluation,\n"
                f"analyze the {prompt_type} of word groups on a precise scale from 0 to 1.\n"
                f"Provide a numerical score first, followed by your explanation."
            )
        
        # 변형 3: 단순화
        if num_variations >= 3:
            variations["variation_3"] = (
                f"Rate the {prompt_type} of word groups from 0-1.\n"
                f"0 = poor, 1 = excellent.\n"
                f"Give score first, then explain."
            )
        
        # 변형 4: 맥락 추가
        if num_variations >= 4:
            variations["variation_4"] = (
                f"{base_prompt}\n\n"
                f"These word groups represent topics extracted from a topic model.\n"
                f"Evaluate their {prompt_type} from 0-1, where higher is better.\n"
                f"Format: <score>\n<explanation>"
            )
        
        # 변형 5: 척도 설명
        if num_variations >= 5:
            variations["variation_5"] = (
                f"{base_prompt}\n\n"
                f"When evaluating {prompt_type}, use this scale:\n"
                f"- 0.0-0.2: Very poor\n"
                f"- 0.2-0.4: Poor\n"
                f"- 0.4-0.6: Average\n"
                f"- 0.6-0.8: Good\n"
                f"- 0.8-1.0: Excellent\n\n"
                f"Provide score first, then explanation."
            )
        
        # 추가 변형 (필요시)
        if num_variations > 5:
            for i in range(6, num_variations + 1):
                variations[f"variation_{i}"] = f"{base_prompt} (Variation {i})"
        
        return variations
    
    def _plot_prompt_variations(self, results: Dict[str, Any], dataset: str):
        """
        프롬프트 변형 분석 시각화
        
        Args:
            results: prompt_variation_analysis 결과
            dataset: 데이터셋 이름
        """
        prompt_type = results['prompt_type']
        variations = list(results['variations'].keys())
        scores = results['scores']
        
        # 유효한 결과만 필터링
        valid_vars = [v for v in variations if scores[v] is not None]
        
        if not valid_vars:
            logger.warning("No valid data for visualization")
            return
        
        # 메트릭별로 그래프 생성
        metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
        
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            
            y = [scores[v][metric] for v in valid_vars]
            x = np.arange(len(valid_vars))
            
            plt.bar(x, y, alpha=0.7)
            plt.xticks(x, [v.replace('variation_', 'V') for v in valid_vars], rotation=45)
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.ylabel('Score')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 통계 정보 표시
            cv = results['statistics']['by_metric'][metric]['cv']
            plt.annotate(f"CV: {cv:.3f}", xy=(0.05, 0.05), xycoords='axes fraction')
        
        plt.suptitle(f"Prompt Variation Analysis: {prompt_type.title()}", fontsize=16)
        plt.tight_layout()
        
        # 저장
        output_path = self.output_dir / f"prompt_variations_{dataset}_{prompt_type}_{self.session_id}.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Prompt variation visualization saved to {output_path}")

    def multi_llm_comparison(self, evaluator_classes: List[Any], llm_names: List[str],
                           dataset: str = "distinct") -> Dict[str, Any]:
        """
        여러 LLM 비교 분석
        
        Args:
            evaluator_classes: LLM evaluator 클래스 리스트 [AnthropicEval, OpenAIEval, ...]
            llm_names: LLM 이름 리스트 ["Anthropic", "OpenAI", ...]
            dataset: 사용할 데이터셋 이름 (기본값: "distinct")
            
        Returns:
            Dict: 분석 결과 딕셔너리
        """
        if len(evaluator_classes) != len(llm_names):
            raise ValueError("evaluator_classes와 llm_names의 길이가 일치해야 합니다")
            
        logger.info(f"Starting multi-LLM comparison with {llm_names} on {dataset} dataset")
        
        # 토픽 데이터 로드
        try:
            topics = self._load_topics(dataset)
            logger.info(f"Loaded {len(topics)} topics from {dataset} dataset")
        except FileNotFoundError as e:
            logger.error(f"Error loading topics: {e}")
            raise
        
        results = {
            'dataset': dataset,
            'llm_names': llm_names,
            'scores': {},
            'statistics': {},
            'pairwise_analysis': {}
        }
        
        # 각 LLM에 대해 평가 수행
        for evaluator_class, llm_name in zip(evaluator_classes, llm_names):
            logger.info(f"Evaluating with {llm_name}")
            
            try:
                # evaluator 인스턴스 생성
                evaluator = evaluator_class()
                
                # 평가 수행
                evaluation_results = evaluator.evaluate_topic_set(topics, f"{dataset.title()} Topics ({llm_name})")
                
                # 결과 저장
                results['scores'][llm_name] = {
                    'coherence': evaluation_results['scores']['coherence'],
                    'distinctiveness': evaluation_results['scores']['distinctiveness'],
                    'diversity': evaluation_results['scores']['diversity'],
                    'semantic_integration': evaluation_results['scores']['semantic_integration'],
                    'overall_score': evaluation_results['scores']['overall_score']
                }
                
                logger.info(f"Evaluation completed for {llm_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating with {llm_name}: {e}")
                results['scores'][llm_name] = None
        
        # LLM 간 비교 분석
        valid_scores = {name: s for name, s in results['scores'].items() if s is not None}
        
        if len(valid_scores) >= 2:
            # 쌍별 상관관계 및 MAD 계산
            llm_names = list(valid_scores.keys())
            n_llms = len(llm_names)
            
            corr_matrix = np.zeros((n_llms, n_llms))
            mad_matrix = np.zeros((n_llms, n_llms))
            
            metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
            
            for i in range(n_llms):
                for j in range(i, n_llms):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        mad_matrix[i, j] = 0.0
                    else:
                        # 모든 메트릭의 점수를 벡터로 변환
                        scores_i = np.array([valid_scores[llm_names[i]][m] for m in metrics])
                        scores_j = np.array([valid_scores[llm_names[j]][m] for m in metrics])
                        
                        # 상관관계 계산
                        corr = np.corrcoef(scores_i, scores_j)[0, 1]
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                        
                        # MAD 계산
                        mad = np.mean(np.abs(scores_i - scores_j))
                        mad_matrix[i, j] = mad
                        mad_matrix[j, i] = mad
            
            # 결과 저장
            results['pairwise_analysis'] = {
                'correlation_matrix': corr_matrix.tolist(),
                'mad_matrix': mad_matrix.tolist(),
                'llm_names': llm_names
            }
            
            # 전체 통계
            mean_corr = np.mean(corr_matrix[np.triu_indices(n_llms, k=1)])
            mean_mad = np.mean(mad_matrix[np.triu_indices(n_llms, k=1)])
            
            results['statistics'] = {
                'mean_correlation': float(mean_corr),
                'mean_mad': float(mean_mad),
                'metrics_analyzed': metrics
            }
            
            # 등급 및 결론
            if mean_corr > 0.9:
                agreement = "very high"
            elif mean_corr > 0.7:
                agreement = "high"
            elif mean_corr > 0.5:
                agreement = "moderate"
            else:
                agreement = "low"
                
            results['statistics']['agreement_grade'] = agreement
            results['statistics']['conclusion'] = f"Inter-model agreement is {agreement} (r={mean_corr:.3f}, MAD={mean_mad:.3f})"
            
            logger.info(f"Multi-LLM comparison completed: {agreement} agreement")
        else:
            logger.error("At least 2 valid evaluation results required for comparison")
            
        # 전체 결과에 저장
        self.results['multi_llm'][dataset] = results
        
        # 시각화 생성 (옵션)
        try:
            self._plot_multi_llm_comparison(results, dataset)
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")
        
        return results
    
    def _plot_multi_llm_comparison(self, results: Dict[str, Any], dataset: str):
        """
        여러 LLM 비교 분석 시각화
        
        Args:
            results: multi_llm_comparison 결과
            dataset: 데이터셋 이름
        """
        llm_names = [name for name in results['llm_names'] if results['scores'].get(name) is not None]
        
        if len(llm_names) < 2:
            logger.warning("At least 2 valid LLMs required for visualization")
            return
        
        # 메트릭별 비교 그래프
        metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
        
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            
            y = [results['scores'][name][metric] for name in llm_names]
            x = np.arange(len(llm_names))
            
            plt.bar(x, y, alpha=0.7)
            plt.xticks(x, llm_names, rotation=45)
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.ylabel('Score')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle(f"Multi-LLM Comparison: {dataset.title()}", fontsize=16)
        plt.tight_layout()
        
        # 저장
        output_path = self.output_dir / f"multi_llm_comparison_{dataset}_{self.session_id}.png"
        plt.savefig(output_path)
        plt.close()
        
        # 상관관계 히트맵
        if 'pairwise_analysis' in results and 'correlation_matrix' in results['pairwise_analysis']:
            plt.figure(figsize=(10, 8))
            
            corr_matrix = np.array(results['pairwise_analysis']['correlation_matrix'])
            
            sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                       xticklabels=llm_names, yticklabels=llm_names)
            
            plt.title(f"Inter-Model Correlation: {dataset.title()}")
            plt.tight_layout()
            
            # 저장
            output_path = self.output_dir / f"multi_llm_correlation_{dataset}_{self.session_id}.png"
            plt.savefig(output_path)
            plt.close()
            
        logger.info(f"Multi-LLM comparison visualizations saved to {self.output_dir}")

    def multi_iteration_stability(self, evaluator_class, dataset: str = "distinct",
                                num_iterations: int = 5) -> Dict[str, Any]:
        """
        반복 실행 안정성 분석
        
        Args:
            evaluator_class: LLM evaluator 클래스 (AnthropicEval, OpenAIEval 등)
            dataset: 사용할 데이터셋 이름 (기본값: "distinct")
            num_iterations: 반복 실행 횟수 (기본값: 5)
            
        Returns:
            Dict: 분석 결과 딕셔너리
        """
        logger.info(f"Starting multi-iteration stability analysis with {num_iterations} iterations on {dataset} dataset")
        
        # 토픽 데이터 로드
        try:
            topics = self._load_topics(dataset)
            logger.info(f"Loaded {len(topics)} topics from {dataset} dataset")
        except FileNotFoundError as e:
            logger.error(f"Error loading topics: {e}")
            raise
        
        results = {
            'dataset': dataset,
            'num_iterations': num_iterations,
            'scores': {},
            'statistics': {}
        }
        
        # 각 반복에 대해 평가 수행
        for i in range(num_iterations):
            logger.info(f"Iteration {i+1}/{num_iterations}")
            
            try:
                # evaluator 인스턴스 생성
                evaluator = evaluator_class()
                
                # 평가 수행
                evaluation_results = evaluator.evaluate_topic_set(topics, f"{dataset.title()} Topics (Iteration {i+1})")
                
                # 결과 저장
                results['scores'][i+1] = {
                    'coherence': evaluation_results['scores']['coherence'],
                    'distinctiveness': evaluation_results['scores']['distinctiveness'],
                    'diversity': evaluation_results['scores']['diversity'],
                    'semantic_integration': evaluation_results['scores']['semantic_integration'],
                    'overall_score': evaluation_results['scores']['overall_score']
                }
                
                logger.info(f"Iteration {i+1} completed")
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                results['scores'][i+1] = None
        
        # 반복 실행 통계 계산
        valid_scores = {i: s for i, s in results['scores'].items() if s is not None}
        
        if valid_scores:
            # 메트릭별 평균, 표준편차, CV, 범위
            metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
            stats_by_metric = {}
            
            for metric in metrics:
                values = [scores[metric] for scores in valid_scores.values()]
                stats_by_metric[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                    'range': max(values) - min(values) if values else 0
                }
            
            # 전체 통계
            results['statistics'] = {
                'by_metric': stats_by_metric,
                'mean_cv': np.mean([s['cv'] for s in stats_by_metric.values()]),
                'mean_range': np.mean([s['range'] for s in stats_by_metric.values()]),
                'metrics_analyzed': metrics
            }
            
            # 등급 및 결론
            cv = results['statistics']['mean_cv']
            if cv < 0.01:
                stability = "very high"
            elif cv < 0.03:
                stability = "high"
            elif cv < 0.07:
                stability = "moderate"
            else:
                stability = "low"
                
            results['statistics']['stability_grade'] = stability
            results['statistics']['conclusion'] = f"Evaluation stability is {stability} (CV={cv:.3f})"
            
            logger.info(f"Multi-iteration stability analysis completed: {stability} stability")
        else:
            logger.error("No valid evaluation results")
            
        # 전체 결과에 저장
        self.results['multi_iteration'][dataset] = results
        
        # 시각화 생성 (옵션)
        try:
            self._plot_multi_iteration_stability(results, dataset)
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")
        
        return results
    
    def _plot_multi_iteration_stability(self, results: Dict[str, Any], dataset: str):
        """
        반복 실행 안정성 분석 시각화
        
        Args:
            results: multi_iteration_stability 결과
            dataset: 데이터셋 이름
        """
        iterations = sorted([i for i in results['scores'].keys() if results['scores'][i] is not None])
        
        if not iterations:
            logger.warning("No valid data for visualization")
            return
        
        # 메트릭별 시계열 그래프
        metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration', 'overall_score']
        
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            
            y = [results['scores'][it][metric] for it in iterations]
            x = iterations
            
            plt.plot(x, y, 'o-', alpha=0.7)
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 통계 정보 표시
            cv = results['statistics']['by_metric'][metric]['cv']
            plt.annotate(f"CV: {cv:.3f}", xy=(0.05, 0.05), xycoords='axes fraction')
            
            # 평균선
            mean = results['statistics']['by_metric'][metric]['mean']
            plt.axhline(y=mean, color='r', linestyle='--', alpha=0.5, label=f'Mean = {mean:.3f}')
            plt.legend()
        
        plt.suptitle(f"Multi-Iteration Stability: {dataset.title()}", fontsize=16)
        plt.tight_layout()
        
        # 저장
        output_path = self.output_dir / f"multi_iteration_stability_{dataset}_{self.session_id}.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Multi-iteration stability visualization saved to {output_path}")

    def propose_mitigation_strategies(self) -> Dict[str, Any]:
        """
        완화 전략 추천
        
        분석 결과를 기반으로 LLM 평가의 한계와 편향을 완화하는 전략 제안
        
        Returns:
            Dict: 추천 전략 딕셔너리
        """
        logger.info("Proposing mitigation strategies based on analysis results")
        
        strategies = {}
        
        # Temperature 민감도 기반 전략
        if self.results['temperature_sensitivity']:
            # 가장 최근의 데이터셋 결과 사용
            dataset = list(self.results['temperature_sensitivity'].keys())[-1]
            temp_results = self.results['temperature_sensitivity'][dataset]
            
            if 'sensitivity_grade' in temp_results.get('statistics', {}):
                sensitivity = temp_results['statistics']['sensitivity_grade']
                cv = temp_results['statistics'].get('mean_cv', float('inf'))
                
                # 전략 선택
                if sensitivity in ['very low', 'low']:
                    suggestion = "Temperature-insensitive (robust)"
                    value = 0.0
                    rationale = "Minimal variation across temperatures, use deterministic (T=0)"
                elif sensitivity == 'moderate':
                    suggestion = "Some temperature sensitivity"
                    value = 0.0
                    rationale = "Moderate variation, use deterministic (T=0) for consistency"
                else:
                    suggestion = "High temperature sensitivity"
                    value = 0.0
                    rationale = "High variation, use deterministic (T=0) and multiple runs"
                
                strategies['temperature'] = {
                    'recommendation': suggestion,
                    'suggested_value': value,
                    'rationale': rationale,
                    'data_support': f"CV={cv:.3f} across temperatures"
                }
        
        # Multi-model 합의 전략
        if self.results['multi_llm']:
            # 가장 최근의 데이터셋 결과 사용
            dataset = list(self.results['multi_llm'].keys())[-1]
            llm_results = self.results['multi_llm'][dataset]
            
            if 'agreement_grade' in llm_results.get('statistics', {}):
                agreement = llm_results['statistics']['agreement_grade']
                corr = llm_results['statistics'].get('mean_correlation', 0)
                
                # 전략 선택
                if agreement in ['very high', 'high']:
                    suggestion = "High inter-model agreement"
                    method = "Use any single model (cost-effective)"
                    rationale = f"Strong correlation (r={corr:.3f}) between models"
                elif agreement == 'moderate':
                    suggestion = "Moderate inter-model agreement"
                    method = "Average 2+ models for important evaluations"
                    rationale = f"Moderate correlation (r={corr:.3f}), combine for robustness"
                else:
                    suggestion = "Low inter-model agreement"
                    method = "Average 3+ models for all evaluations"
                    rationale = f"Weak correlation (r={corr:.3f}), combine multiple models"
                
                strategies['multi_model'] = {
                    'recommendation': suggestion,
                    'method': method,
                    'rationale': rationale,
                    'data_support': f"Mean correlation r={corr:.3f}"
                }
        
        # 앙상블 방법 (종합)
        strategies['ensemble_methods'] = {
            'recommendation': "Recommended",
            'methods': [
                "Temperature ensemble: Average scores from T∈{0, 0.3, 0.7}",
                "Model ensemble: Average scores from Anthropic + OpenAI + Gemini",
                "Prompt ensemble: Average scores from 3-5 prompt variations",
                "Iteration ensemble: Average scores from 5+ independent runs"
            ],
            'expected_improvement': "15-30% reduction in score variance"
        }
        
        # 결론
        strategies['conclusions'] = [
            "Bias Mitigation: Use ensemble methods to reduce single-model bias",
            "Hallucination Risk: Temperature=0 minimizes hallucination",
            "Score Stability: Multi-iteration averaging improves reliability",
            "Best Practice: Combine 2+ LLMs with prompt ensemble for critical evaluations"
        ]
        
        # 저장
        self.results['mitigation_strategies'] = strategies
        
        logger.info("Mitigation strategies proposed")
        return strategies

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        종합 분석 리포트 생성
        
        Args:
            output_path: 리포트 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            str: 마크다운 형식의 리포트
        """
        logger.info("Generating comprehensive analysis report")
        
        report = [
            "# LLM Robustness Analysis Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # 1. Temperature 민감도 분석
        if self.results['temperature_sensitivity']:
            report.append("## 1. Temperature Sensitivity Analysis")
            report.append("")
            
            # 가장 최근의 데이터셋 결과 사용
            dataset = list(self.results['temperature_sensitivity'].keys())[-1]
            temp_results = self.results['temperature_sensitivity'][dataset]
            
            temperatures = temp_results['temperatures']
            report.append(f"**Temperatures tested**: {temperatures}")
            report.append("")
            report.append("### Results")
            report.append("")
            
            # 결과 테이블
            report.append("| Temperature | Mean Score | Std Dev | CV |")
            report.append("|-------------|------------|---------|------|")
            
            for temp in temperatures:
                if temp_results['scores'].get(temp) is not None:
                    overall_score = temp_results['scores'][temp]['overall_score']
                    
                    # 각 메트릭의 평균 계산
                    metric_values = [temp_results['scores'][temp][m] for m in 
                                   ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']]
                    mean_score = np.mean(metric_values)
                    std_dev = np.std(metric_values)
                    cv = std_dev / mean_score if mean_score > 0 else 0
                    
                    report.append(f"| {temp} | {mean_score:.3f} | {std_dev:.3f} | {cv:.3f} |")
            
            report.append("")
            
            # 종합 통계
            if 'statistics' in temp_results:
                mean_range = temp_results['statistics'].get('mean_range', 0)
                mean_cv = temp_results['statistics'].get('mean_cv', 0)
                sensitivity = temp_results['statistics'].get('sensitivity_grade', 'unknown')
                
                report.append(f"**Mean Range**: {mean_range:.3f} ({sensitivity} sensitivity {'✓' if sensitivity in ['very low', 'low'] else '⚠'})")
                report.append(f"**Mean CV**: {mean_cv:.3f}")
            
            report.append("")
        
        # 2. 프롬프트 변형 분석
        if self.results['prompt_variations']:
            report.append("## 2. Prompt Variation Analysis")
            report.append("")
            
            # 가장 최근의 데이터셋 결과 사용
            dataset = list(self.results['prompt_variations'].keys())[-1]
            prompt_results = self.results['prompt_variations'][dataset]
            
            variations = list(prompt_results['variations'].keys())
            report.append(f"**Variations tested**: {len(variations)}")
            report.append("")
            
            # 결과 테이블
            report.append("| Variation | Mean Score | Std Dev | CV |")
            report.append("|-----------|------------|---------|------|")
            
            for var in variations:
                if prompt_results['scores'].get(var) is not None:
                    overall_score = prompt_results['scores'][var]['overall_score']
                    
                    # 각 메트릭의 평균 계산
                    metric_values = [prompt_results['scores'][var][m] for m in 
                                   ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']]
                    mean_score = np.mean(metric_values)
                    std_dev = np.std(metric_values)
                    cv = std_dev / mean_score if mean_score > 0 else 0
                    
                    report.append(f"| {var} | {mean_score:.3f} | {std_dev:.3f} | {cv:.3f} |")
            
            report.append("")
            
            # 종합 통계
            if 'statistics' in prompt_results:
                mean_range = prompt_results['statistics'].get('mean_range', 0)
                mean_cv = prompt_results['statistics'].get('mean_cv', 0)
                sensitivity = prompt_results['statistics'].get('sensitivity_grade', 'unknown')
                
                report.append(f"**Mean Range**: {mean_range:.3f} ({sensitivity} stability {'✓' if sensitivity in ['very low', 'low'] else '⚠'})")
                report.append(f"**Mean CV**: {mean_cv:.3f}")
            
            report.append("")
        
        # 3. 멀티 LLM 비교
        if self.results['multi_llm']:
            report.append("## 3. Multi-LLM Comparison")
            report.append("")
            
            # 가장 최근의 데이터셋 결과 사용
            dataset = list(self.results['multi_llm'].keys())[-1]
            llm_results = self.results['multi_llm'][dataset]
            
            llm_names = [name for name in llm_results['llm_names'] 
                       if llm_results['scores'].get(name) is not None]
            report.append(f"**Models tested**: {', '.join(llm_names)}")
            report.append("")
            
            # 모델 간 일치도
            if 'statistics' in llm_results:
                corr = llm_results['statistics'].get('mean_correlation', 0)
                mad = llm_results['statistics'].get('mean_mad', 0)
                
                report.append("### Inter-Model Agreement")
                report.append("")
                report.append(f"- **Mean Pairwise Correlation**: {corr:.3f} ({'high ✓' if corr > 0.7 else 'moderate'})")
                report.append(f"- **Mean Topic Variance**: {mad:.3f} ({'low ✓' if mad < 0.1 else 'high'})")
            
            report.append("")
        
        # 4. 완화 전략
        if self.results['mitigation_strategies']:
            report.append("## 4. Mitigation Strategies")
            report.append("")
            
            strategies = self.results['mitigation_strategies']
            
            # Temperature
            if 'temperature' in strategies:
                temp_strategy = strategies['temperature']
                report.append("### Temperature")
                report.append("")
                report.append(f"- **Recommendation**: {temp_strategy['recommendation']}")
                report.append(f"- **Suggested Value**: {temp_strategy['suggested_value']}")
                report.append(f"- **Rationale**: {temp_strategy['rationale']}")
                report.append("")
            
            # Multi-model
            if 'multi_model' in strategies:
                model_strategy = strategies['multi_model']
                report.append("### Multi-Model Consensus")
                report.append("")
                report.append(f"- **Recommendation**: {model_strategy['recommendation']}")
                report.append(f"- **Method**: {model_strategy['method']}")
                report.append(f"- **Rationale**: {model_strategy['rationale']}")
                report.append("")
            
            # 앙상블 방법
            if 'ensemble_methods' in strategies:
                ensemble = strategies['ensemble_methods']
                report.append("### Ensemble Methods (Recommended)")
                report.append("")
                for method in ensemble['methods']:
                    report.append(f"- {method}")
                report.append("")
                report.append(f"**Expected Improvement**: {ensemble['expected_improvement']}")
            
            report.append("")
            
            # 결론
            if 'conclusions' in strategies:
                report.append("## Conclusions")
                report.append("")
                for i, conclusion in enumerate(strategies['conclusions'], 1):
                    report.append(f"{i}. **{conclusion}**")
                report.append("")
        
        # 저장
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")
        
        return report_text

    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        분석 결과 저장
        
        Args:
            output_path: 저장 경로 (None이면 기본 경로 사용)
            
        Returns:
            str: 저장 경로
        """
        if output_path is None:
            output_path = self.output_dir / f"robustness_results_{self.session_id}.pkl"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)


# 예시 실행 코드 (스크립트로 직접 실행하는 경우)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Robustness Analysis")
    parser.add_argument("--analysis", choices=["temperature", "prompt", "multi-llm", "multi-iteration", "all"],
                      default="all", help="Analysis type to run")
    parser.add_argument("--model", choices=["anthropic", "openai", "grok", "all"],
                      default="anthropic", help="LLM model to use")
    parser.add_argument("--dataset", choices=["distinct", "similar", "more_similar"],
                      default="distinct", help="Dataset to analyze")
    parser.add_argument("--output-dir", type=str, default=None, 
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # 분석기 초기화
    analyzer = RobustnessAnalyzer(output_dir=args.output_dir)
    
    # 모델 선택
    try:
        if args.model == "anthropic":
            from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicEval
            evaluator_class = AnthropicEval
        elif args.model == "openai":
            from topic_llm.openai_topic_evaluator import TopicEvaluatorLLM as OpenAIEval
            evaluator_class = OpenAIEval
        elif args.model == "grok":
            from topic_llm.grok_topic_evaluator import TopicEvaluatorLLM as GrokEval
            evaluator_class = GrokEval
        else:
            from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicEval
            evaluator_class = AnthropicEval
    except ImportError as e:
        logger.error(f"Error importing evaluator: {e}")
        logger.error("Please ensure the LLM evaluators are correctly implemented")
        import sys
        sys.exit(1)
    
    # 분석 실행
    if args.analysis == "temperature" or args.analysis == "all":
        analyzer.temperature_sensitivity_analysis(evaluator_class, dataset=args.dataset)
    
    if args.analysis == "prompt" or args.analysis == "all":
        analyzer.prompt_variation_analysis(evaluator_class, dataset=args.dataset, num_variations=3)
    
    if args.analysis == "multi-llm" or args.analysis == "all":
        if args.model == "all":
            try:
                from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicEval
                from topic_llm.openai_topic_evaluator import TopicEvaluatorLLM as OpenAIEval
                
                evaluator_classes = [AnthropicEval, OpenAIEval]
                llm_names = ["Anthropic", "OpenAI"]
                
                try:
                    from topic_llm.grok_topic_evaluator import TopicEvaluatorLLM as GrokEval
                    evaluator_classes.append(GrokEval)
                    llm_names.append("Grok")
                except ImportError:
                    logger.warning("Grok evaluator not available")
                
                analyzer.multi_llm_comparison(evaluator_classes, llm_names, dataset=args.dataset)
            except ImportError as e:
                logger.error(f"Error importing evaluators for multi-LLM comparison: {e}")
        else:
            logger.warning("Multi-LLM comparison requires --model=all")
    
    if args.analysis == "multi-iteration" or args.analysis == "all":
        analyzer.multi_iteration_stability(evaluator_class, dataset=args.dataset, num_iterations=3)
    
    # 완화 전략 추천
    analyzer.propose_mitigation_strategies()
    
    # 리포트 생성
    report_path = analyzer.output_dir / f"robustness_report_{args.dataset}_{analyzer.session_id}.md"
    analyzer.generate_report(output_path=str(report_path))
    
    # 결과 저장
    analyzer.save_results()
    
    logger.info(f"Analysis complete. Report saved to: {report_path}")
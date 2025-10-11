# 필요한 라이브러리 임포트
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Union
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import traceback
from scipy.stats import wasserstein_distance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
import os

class TopicModelStatEvaluator:
    def __init__(self, device=None, cache_paths: Dict[str, str] = None):
        """Initialize statistical topic model evaluator"""
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_stats = {}
        self.topic_sizes = {}
        self.vocabulary_size = 0
        self.total_documents = 0
        self.natural_distribution = None
        self.natural_clusters = None
        
        # Setup cache paths
        self.cache_paths = cache_paths or {
            'base_cache_dir': os.path.join('cache', 'default'),
            'stats_cache': os.path.join('cache', 'default', 'evaluation_stats.pkl')
        }
        
        os.makedirs(self.cache_paths['base_cache_dir'], exist_ok=True)

    # 1. 일관성(Coherence) 평가
    def _calculate_coherence(self, topics: List[List[str]]) -> float:
        """Calculate topic coherence"""
        try:
            if not topics:
                return 0.0
           
      
            word_doc_freq = self.model_stats.get('word_doc_freq', {})
            co_doc_freq = self.model_stats.get('co_doc_freq', {})
            total_docs = self.total_documents or len(word_doc_freq)
            
            coherence_scores = []
            for topic_words in topics:
        
                if len(topic_words) < 2:
                    continue
                    
                # Use only top 10 keywords
                topic_words = topic_words[:10]
          
                topic_coherence = []
                for i in range(1, len(topic_words)):
                    for j in range(0, i):
                        word2, word1 = topic_words[i], topic_words[j]
                    
     
                        d_w1 = word_doc_freq.get(word1, 0)
                        if d_w1 == 0:
                            continue
              
               
                        pair = (word1, word2) if word1 < word2 else (word2, word1)
                        d_w1w2 = co_doc_freq.get(pair, 0)
                        
                        # PMI 계산
                        p_w1 = d_w1 / total_docs
                        p_w2 = word_doc_freq.get(word2, 0) / total_docs
                        p_w1w2 = d_w1w2 / total_docs
                        
                        if p_w1w2 == 0:
                            continue
                            
                        pmi = np.log(p_w1w2 / (p_w1 * p_w2))
                        
                        # NPMI 계산 및 정규화
                        npmi = pmi / (-np.log(p_w1w2))
                        normalized_npmi = (npmi + 1) / 2
                        topic_coherence.append(normalized_npmi)
            
                if topic_coherence:
                    coherence_scores.append(np.mean(topic_coherence))
        
            if coherence_scores:
                return np.mean(coherence_scores)
            return 0.0
        
        except Exception as e:
            print(f"[WARNING] Error calculating coherence: {str(e)}")
            return 0.0

    # 2. 구별성(Jensen-Shannon Divergence) 평가
    def _calculate_jsd(self, topics: List[List[str]], topic_assignments: List[int], **kwargs) -> float:
        """Jensen-Shannon Divergence 계산"""
        valid_assignments = [t for t in topic_assignments if t >= 0]
        if not valid_assignments:
            return 0.0
        
        topic_dist = np.bincount(valid_assignments, minlength=len(topics))
        if np.sum(topic_dist) == 0:
            return 0.0
        
        topic_dist = topic_dist / np.sum(topic_dist)
        uniform_dist = np.ones_like(topic_dist) / len(topic_dist)
    
        return jensenshannon(topic_dist, uniform_dist)

    def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        """토픽 모델 종합 평가"""
        try:
            self._validate_inputs(topics, docs, topic_assignments)
            
            # 각 지표의 원점수 계산 (기존 메서드 사용)
            coherence_score = self._calculate_coherence(topics)
            distinctiveness_score = self._calculate_jsd(topics, topic_assignments) # Jensen-Shannon Divergence 사용
            
            # 결과를 새로운 형식으로 구성
            coherence_result = {
                'average_coherence': coherence_score,
                'topic_coherence': [coherence_score]  # 개별 토픽 점수가 필요하다면 수정 필요
            }
            
            distinctiveness_result = {
                'average_distinctiveness': distinctiveness_score,
                'topic_distinctiveness': [distinctiveness_score]  # 개별 토픽 점수가 필요하다면 수정 필요
            }
            
            # 고정 가중치 정의
            weights = {
                'coherence': 0.7,        # 토픽 품질
                'distinctiveness': 0.3   # 토픽 간 구별성
            }
            
            # 원점수 
            raw_scores = {
                'coherence': coherence_score,
                'distinctiveness': distinctiveness_score,
            }
            
            # 가중치 적용된 개별 점수
            weighted_scores = {
                'coherence': raw_scores['coherence'] * weights['coherence'],
                'distinctiveness': raw_scores['distinctiveness'] * weights['distinctiveness'],
            }
            
            # 최종 종합 점수
            overall_score = sum(weighted_scores.values())
            
            return {
                'coherence': coherence_result,
                'distinctiveness': distinctiveness_result,
                'raw_scores': raw_scores,
                'weighted_scores': weighted_scores,
                'weights': weights,
                'overall_score': overall_score
            }
            
        except Exception as e:
            print(f"[ERROR] 토픽 모델 평가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {
                'coherence': self._get_default_evaluation_result(),
                'distinctiveness': self._get_default_evaluation_result(),
                'raw_scores': {'coherence': 0.0, 'distinctiveness': 0.0},
                'weighted_scores': {'coherence': 0.0, 'distinctiveness': 0.0},
                'weights': {'coherence': 0.7, 'distinctiveness': 0.3},
                'overall_score': 0.0
            }

    # 핵심 유틸리티 메서드
    def set_model_stats(self, **kwargs):
        """모델 통계 정보 설정"""
        try:
            self.topic_sizes = kwargs.get('topic_sizes', {})
            self.vocabulary_size = kwargs.get('vocabulary_size', 0)
            self.total_documents = kwargs.get('total_documents', 0)
            self.model_stats = kwargs
            
        except Exception as e:
            print(f"[WARNING] 통계 정보 설정 중 오류 발생: {str(e)}")

    # 데이터 검증 및 정규화 메서드
    def _validate_inputs(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> None:
        """Validate input data"""
        if not topics or not all(isinstance(topic, list) for topic in topics):
            raise ValueError("Invalid topic format")
    
        if not docs:
            raise ValueError("Documents are empty")
    
        if len(docs) != len(topic_assignments):
            raise ValueError("Number of documents does not match topic assignments")

    def _get_default_evaluation_result(self) -> Dict[str, float]:
        """Get default evaluation result"""
        return {
            'average_coherence': 0.0,
            'topic_coherence': []
        }

    def calculate_coherence(self, topics: List[List[str]], word_doc_freq: Dict[str, int], co_doc_freq: Dict[Tuple[str, str], int]) -> float:
        print("[DEBUG] Calculating coherence...")
        print(f"[DEBUG] Number of topics: {len(topics)}")
        print(f"[DEBUG] word_doc_freq size: {len(word_doc_freq)}")
        print(f"[DEBUG] co_doc_freq size: {len(co_doc_freq)}")
        
        if not topics or not word_doc_freq or not co_doc_freq:
            print("[DEBUG] Missing data for coherence calculation")
            return 0.0
        
        coherence_scores = []
        for topic in topics:
            if len(topic) < 2:  # 키워드가 2개 미만인 경우
                print(f"[DEBUG] Topic has less than 2 keywords: {topic}")
                continue
            
            topic_score = 0
            pairs = 0
            
            for i, word1 in enumerate(topic):
                for word2 in topic[i+1:]:
                    # 디버깅을 위한 출력
                    pair = tuple(sorted([word1, word2]))
                    print(f"[DEBUG] Processing pair: {pair}")
                    print(f"[DEBUG] word1 freq: {word_doc_freq.get(word1, 0)}")
                    print(f"[DEBUG] word2 freq: {word_doc_freq.get(word2, 0)}")
                    print(f"[DEBUG] co-occurrence freq: {co_doc_freq.get(pair, 0)}")
                    
                    if word1 not in word_doc_freq or word2 not in word_doc_freq:
                        print(f"[DEBUG] Missing word frequency for {word1} or {word2}")
                        continue
                    
                    if pair not in co_doc_freq:
                        print(f"[DEBUG] Missing co-occurrence frequency for pair {pair}")
                        continue
                    
                    # 실제 계산
                    score = np.log((co_doc_freq[pair] + 1) / word_doc_freq[word1])
                    topic_score += score
                    pairs += 1
                    
            if pairs > 0:
                coherence_scores.append(topic_score / pairs)
            else:
                print(f"[DEBUG] No valid pairs found for topic")
        
        if not coherence_scores:
            print("[DEBUG] No coherence scores calculated")
            return 0.0
        
        final_coherence = np.mean(coherence_scores)
        print(f"[DEBUG] Final coherence score: {final_coherence}")
        return final_coherence





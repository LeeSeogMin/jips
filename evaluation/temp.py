# 토픽의 다면적 특성을 종합적으로 고려 1차 수정

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Union, Tuple
from scipy.stats import entropy, zscore
from scipy.spatial.distance import jensenshannon, pdist, squareform
from sklearn.cluster import DBSCAN
import traceback
import math

class TopicModelNeuralEvaluator:
    def __init__(self, 
                 model_embeddings: Dict[str, torch.Tensor], 
                 embedding_dim: int = 768, 
                 device: str = 'cpu'):
        self.model_embeddings = model_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.missing_embeddings = set()

    def _analyze_local_structure(self, topic_keywords: List[str]) -> Dict[str, float]:
        """토픽 내부 구조 분석"""
        # 키워드 임베딩 획득
        embeddings = []
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            embeddings.append(emb.cpu().numpy())
        
        embeddings = np.stack(embeddings)
        
        # 1. 키워드 간 거리 행렬 계산
        distances = squareform(pdist(embeddings, metric='cosine'))
        
        # 2. 군집 분석
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        noise_ratio = sum(1 for l in clustering.labels_ if l == -1) / len(clustering.labels_)
        
        # 3. 거리 분포 분석
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        skewness = np.mean(((distances - avg_distance) / std_distance) ** 3)
        
        return {
            'cluster_count': n_clusters,
            'noise_ratio': noise_ratio,
            'avg_distance': float(avg_distance),
            'distance_std': float(std_distance),
            'distance_skewness': float(skewness)
        }

    def _measure_semantic_scatter(self, topic_keywords: List[str]) -> float:
        """의미적 산포도 측정"""
        embeddings = []
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            embeddings.append(emb)
        
        embeddings_tensor = torch.stack(embeddings)
        centroid = torch.mean(embeddings_tensor, dim=0)
        
        # 중심으로부터의 거리 계산
        distances = F.cosine_similarity(embeddings_tensor, centroid.unsqueeze(0))
        
        # 거리 분포의 특성 분석
        mean_dist = torch.mean(distances)
        std_dist = torch.std(distances)
        cv = std_dist / (mean_dist + 1e-10)  # 변동계수
        
        # 극단치 비율 계산
        outliers = torch.sum((distances - mean_dist).abs() > 2 * std_dist)
        outlier_ratio = outliers.float() / len(distances)
        
        scatter_score = (cv + outlier_ratio) / 2
        return float(min(1.0, scatter_score))

    def _detect_isolated_subgroups(self, topic_keywords: List[str]) -> float:
        """고립된 하위그룹 탐지"""
        embeddings = []
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            embeddings.append(emb.cpu().numpy())
        
        embeddings = np.stack(embeddings)
        
        # DBSCAN으로 하위그룹 탐지
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        labels = clustering.labels_
        
        if len(set(labels)) <= 1:  # 하위그룹이 없거나 하나만 있는 경우
            return 0.0
            
        # 군집 간 거리 계산
        cluster_centroids = []
        for label in set(labels):
            if label == -1:  # 노이즈 포인트 제외
                continue
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            cluster_centroids.append(centroid)
            
        if len(cluster_centroids) <= 1:
            return 0.0
            
        # 군집 간 평균 거리 계산
        cluster_centroids = np.stack(cluster_centroids)
        between_distances = pdist(cluster_centroids, metric='cosine')
        isolation_score = float(np.mean(between_distances))
        
        return min(1.0, isolation_score)

    def _calculate_outlier_ratio(self, topic_keywords: List[str]) -> float:
        """이상치 비율 계산"""
        embeddings = []
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            embeddings.append(emb)
        
        embeddings_tensor = torch.stack(embeddings)
        
        # 페어와이즈 거리 계산
        n = len(embeddings)
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = 1 - F.cosine_similarity(
                    embeddings_tensor[i].unsqueeze(0), 
                    embeddings_tensor[j].unsqueeze(0)
                )
                distances[i,j] = distances[j,i] = dist
        
        # 각 포인트의 평균 거리 계산
        mean_distances = torch.mean(distances, dim=1)
        
        # Modified Z-score 방법으로 이상치 탐지
        median_dist = torch.median(mean_distances)
        mad = torch.median(torch.abs(mean_distances - median_dist))
        modified_zscores = 0.6745 * (mean_distances - median_dist) / (mad + 1e-10)
        
        outliers = torch.sum(modified_zscores > 3.5)
        ratio = float(outliers / len(topic_keywords))
        
        return min(1.0, ratio)

    def evaluate_topic_coherence(self, topics: List[List[str]]) -> Dict[str, Any]:
        """개선된 토픽 일관성 평가"""
        coherence_scores = []
        for topic_keywords in topics:
            if len(topic_keywords) < 2:
                coherence_scores.append(0.0)
                continue

            # 다면 평가 수행
            local_structure = self._analyze_local_structure(topic_keywords)
            semantic_scatter = self._measure_semantic_scatter(topic_keywords)
            outlier_ratio = self._calculate_outlier_ratio(topic_keywords)
            
            # 기본 일관성 점수 계산
            base_coherence = 1.0 - (
                0.4 * semantic_scatter +
                0.3 * local_structure['noise_ratio'] +
                0.3 * outlier_ratio
            )
            
            # 구조적 패널티 적용
            structural_penalty = (
                0.2 * (local_structure['cluster_count'] - 1) / len(topic_keywords) +
                0.3 * local_structure['distance_std'] +
                0.2 * abs(local_structure['distance_skewness'])
            )
            
            final_coherence = base_coherence * (1.0 - min(0.5, structural_penalty))
            coherence_scores.append(float(final_coherence))

        return {
            'topic_coherence': coherence_scores,
            'average_coherence': float(np.mean(coherence_scores))
        }

    def evaluate_topic_distinctiveness(self, topics: List[List[str]]) -> Dict[str, Any]:
        """개선된 토픽 구별성 평가"""
        if not topics:
            return {'topic_distinctiveness': [], 'average_distinctiveness': 0.0}
            
        distinctiveness_scores = []
        quality_metrics = []

        # 각 토픽의 품질 메트릭 계산
        for topic_keywords in topics:
            scatter = self._measure_semantic_scatter(topic_keywords)
            isolation = self._detect_isolated_subgroups(topic_keywords)
            quality = 1.0 - (0.7 * scatter + 0.3 * isolation)
            quality_metrics.append(quality)

        # 토픽 간 구별성 계산
        for i, topic1 in enumerate(topics):
            topic_scores = []
            for j, topic2 in enumerate(topics):
                if i != j:
                    # 기본 구별성
                    base_distinctiveness = self._calculate_semantic_overlap(topic1, topic2)
                    
                    # 품질 기반 조정
                    quality_factor = min(quality_metrics[i], quality_metrics[j])
                    adjusted_distinctiveness = base_distinctiveness * quality_factor
                    
                    topic_scores.append(adjusted_distinctiveness)

            if topic_scores:
                distinctiveness_scores.append(float(np.mean(topic_scores)))

        return {
            'topic_distinctiveness': distinctiveness_scores,
            'average_distinctiveness': float(np.mean(distinctiveness_scores))
        }

    def _calculate_semantic_overlap(self, topic1: List[str], topic2: List[str]) -> float:
        """개선된 의미적 중복도 계산"""
        # 토픽 임베딩 계산
        emb1 = self._get_topic_embedding(topic1[:10], use_weights=True)
        emb2 = self._get_topic_embedding(topic2[:10], use_weights=True)
        
        # 의미적 유사도
        semantic_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        
        # 키워드 중복
        common_keywords = set(topic1) & set(topic2)
        word_overlap = len(common_keywords) / min(len(topic1), len(topic2))
        
        # 토픽 품질 반영
        quality1 = 1.0 - self._measure_semantic_scatter(topic1)
        quality2 = 1.0 - self._measure_semantic_scatter(topic2)
        quality_factor = min(quality1, quality2)
        
        overlap_score = (
            0.5 * semantic_sim.item() +
            0.5 * word_overlap
        ) * quality_factor
        
        distinctiveness = 1.0 - overlap_score
        return float(distinctiveness)

    def evaluate(self, topics: List[List[str]], docs: List[str]) -> Dict[str, Any]:
        """통합 평가 수행"""
        try:
            coherence_result = self.evaluate_topic_coherence(topics)
            distinctiveness_result = self.evaluate_topic_distinctiveness(topics)
            
            raw_scores = {
                'coherence': coherence_result['average_coherence'],
                'distinctiveness': distinctiveness_result['average_distinctiveness']
            }
            
            # 토픽 품질에 기반한 가중치 조정
            coherence_weight = 0.7  # 일관성에 더 큰 가중치
            distinctiveness_weight = 0.3
            
            weighted_scores = {
                'coherence': raw_scores['coherence'] * coherence_weight,
                'distinctiveness': raw_scores['distinctiveness'] * distinctiveness_weight
            }
            
            overall_score = weighted_scores['coherence'] + weighted_scores['distinctiveness']

            return {
                'raw_scores': raw_scores,
                'weighted_scores': weighted_scores,
                'overall_score': overall_score,
                'topic_coherence': coherence_result['topic_coherence'],
                'topic_distinctiveness': distinctiveness_result['topic_distinctiveness']
            }

        except Exception as e:
            print(f"[ERROR] 평가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {}
        
    def _get_embedding(self, keyword: str) -> torch.Tensor:
        """키워드 임베딩 반환"""
        try:
            if keyword in self.model_embeddings:
                emb = self.model_embeddings[keyword]
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                if emb.dtype != torch.float32:
                    emb = emb.to(torch.float32)
                return emb.to(self.device)

            if self.embedding_dim is None:
                raise ValueError("임베딩 차원이 설정되지 않았습니다.")

            self.missing_embeddings.add(keyword)
            default_emb = torch.randn(self.embedding_dim, device=self.device, dtype=torch.float32)
            default_emb = F.normalize(default_emb, p=2, dim=0)
            self.model_embeddings[keyword] = default_emb
            return default_emb

        except Exception as e:
            print(f"[ERROR] 임베딩 변환 중 오류 발생: {str(e)}")
            return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)

    def _calculate_term_weights(self, keywords: List[str]) -> torch.Tensor:
        """계층적 키워드 가중치 계산"""
        position_weights = torch.tensor([
            1.1, 1.05, 1.0, 0.95, 0.9,     # 최상위 키워드
            0.85, 0.8, 0.75, 0.7, 0.65,    # 중위 키워드
            0.6, 0.55, 0.5, 0.45, 0.4      # 하위 키워드
        ], device=self.device)

        keywords = keywords[:15]
        weights = position_weights[:len(keywords)]
        return F.softmax(weights, dim=0)

    def _get_topic_embedding(self, topic_keywords: List[str], use_weights: bool = True) -> torch.Tensor:
        """토픽 임베딩 계산"""
        if not topic_keywords:
            return torch.zeros(self.embedding_dim, device=self.device)

        keyword_embeddings = []
        valid_keywords = []

        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            if not torch.any(torch.isnan(emb)):
                keyword_embeddings.append(emb)
                valid_keywords.append(keyword)

        if not keyword_embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)

        if use_weights:
            weights = self._calculate_term_weights(valid_keywords)
        else:
            weights = torch.ones(len(keyword_embeddings), device=self.device) / len(keyword_embeddings)

        embeddings_tensor = torch.stack(keyword_embeddings)
        weighted_sum = torch.sum(embeddings_tensor * weights.unsqueeze(1), dim=0)

        return F.normalize(weighted_sum, p=2, dim=0)

    def set_model_embeddings(self, embeddings: Dict[str, torch.Tensor], embedding_dim: int):
        """Set model embeddings for evaluation"""
        self.model_embeddings = embeddings
        self.embedding_dim = embedding_dim

# 일관성, 구별성 점수 완화한 코드 

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Union, Tuple
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import traceback
import math
import os

class TopicModelNeuralEvaluator:
    def __init__(self, 
                 model_embeddings: Dict[str, torch.Tensor], 
                 embedding_dim: int = 768, 
                 device: str = 'cpu'):
        """Initialize neural topic model evaluator"""
        self.model_embeddings = model_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.missing_embeddings = set()

    def _get_embedding(self, keyword: str) -> torch.Tensor:
        """키워드 임베딩 반환"""
        try:
            if keyword in self.model_embeddings:
                emb = self.model_embeddings[keyword]
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                if emb.dtype != torch.float32:
                    emb = emb.to(torch.float32)
                return emb.to(self.device)

            if self.embedding_dim is None:
                raise ValueError("임베딩 차원이 설정되지 않았습니다.")

            self.missing_embeddings.add(keyword)
            default_emb = torch.randn(self.embedding_dim, device=self.device, dtype=torch.float32)
            default_emb = F.normalize(default_emb, p=2, dim=0)
            self.model_embeddings[keyword] = default_emb
            return default_emb

        except Exception as e:
            print(f"[ERROR] 임베딩 변환 중 오류 발생: {str(e)}")
            return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)

    def _calculate_term_weights(self, keywords: List[str]) -> torch.Tensor:
        """계층적 키워드 가중치 계산"""
        # 가중치를 더 완만하게 조정
        position_weights = torch.tensor([
            1.1, 1.05, 1.0, 0.95, 0.9,    # 최상위 키워드
            0.85, 0.8, 0.75, 0.7, 0.65,   # 중위 키워드
            0.6, 0.55, 0.5, 0.45, 0.4     # 하위 키워드
        ], device=self.device)

        keywords = keywords[:15]
        weights = position_weights[:len(keywords)]
        return F.softmax(weights, dim=0)

    def _get_topic_embedding(self, topic_keywords: List[str], use_weights: bool = True) -> torch.Tensor:
        """토픽 임베딩 계산"""
        if not topic_keywords:
            return torch.zeros(self.embedding_dim, device=self.device)

        keyword_embeddings = []
        valid_keywords = []

        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            if not torch.any(torch.isnan(emb)):
                keyword_embeddings.append(emb)
                valid_keywords.append(keyword)

        if not keyword_embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)

        if use_weights:
            weights = self._calculate_term_weights(valid_keywords)
        else:
            weights = torch.ones(len(keyword_embeddings), device=self.device) / len(keyword_embeddings)

        embeddings_tensor = torch.stack(keyword_embeddings)
        weighted_sum = torch.sum(embeddings_tensor * weights.unsqueeze(1), dim=0)

        return F.normalize(weighted_sum, p=2, dim=0)

    def evaluate_topic_coherence(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 일관성 평가 - 개선된 버전"""
        coherence_scores = []
        position_weights = self._calculate_term_weights(['dummy'] * 10)

        for topic_keywords in topics:
            if not topic_keywords:
                coherence_scores.append(0.0)
                continue

            top_keywords = topic_keywords[:10]
            vectors = []

            for keyword in top_keywords:
                vec = self._get_embedding(keyword)
                if not torch.any(torch.isnan(vec)):
                    vectors.append(vec)

            if len(vectors) < 2:
                coherence_scores.append(0.0)
                continue

            vectors = torch.stack(vectors)

            # 핵심 키워드 응집도 - 가중치 조정
            core_coherence = 0.0
            core_pairs = 0
            for i in range(min(5, len(vectors))):
                for j in range(i+1, min(5, len(vectors))):
                    sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                    weight = position_weights[i] * position_weights[j] * 1.5  # 가중치 감소
                    core_coherence += sim.item() * weight
                    core_pairs += weight

            core_coherence = core_coherence / core_pairs if core_pairs > 0 else 0.0

            # 전체 키워드 응집도 - 가중치 조정
            full_coherence = 0.0
            total_weight = 0.0
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    weight = position_weights[i] * position_weights[j]
                    sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                    full_coherence += sim.item() * weight
                    total_weight += weight

            full_coherence = full_coherence / total_weight if total_weight > 0 else 0.0

            # 주제 명확성 - 보정 계수 추가
            core_vectors = vectors[:5]
            center = torch.mean(core_vectors, dim=0)
            topic_clarity = torch.mean(F.cosine_similarity(core_vectors, center.unsqueeze(0)))
            clarity_boost = 1.2  # 명확성 점수 향상을 위한 보정

            # 최종 점수 계산 방식 개선
            raw_score = (
                0.4 * core_coherence +    # 핵심 키워드 응집도 비중 감소
                0.3 * full_coherence +    # 전체 키워드 응집도 
                0.3 * (topic_clarity * clarity_boost)  # 주제 명확성 비중 증가
            )

            # 시그모이드 함수 파라미터 조정
            transformed_score = 1 / (1 + math.exp(-4 * (raw_score - 0.3)))  # 중간점 하향 조정
            final_score = 0.1 + 0.85 * transformed_score  # 점수 범위 확대

            # 점수 보정
            if final_score > 0.6:
                final_score = min(0.95, final_score * 1.1)
            elif final_score < 0.2:
                final_score = max(0.1, final_score * 0.9)

            coherence_scores.append(float(final_score))

        return {
            'topic_coherence': coherence_scores,
            'average_coherence': float(np.mean(coherence_scores)) if coherence_scores else 0.0
        }

    def evaluate_topic_distinctiveness(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 간 구별성 평가 - 개선된 버전"""
        if not topics:
            return {'topic_distinctiveness': [], 'average_distinctiveness': 0.0}

        distinctiveness_scores = []

        for i, topic1 in enumerate(topics):
            topic_scores = []
            for j, topic2 in enumerate(topics):
                if i != j:
                    distinctiveness = self._calculate_semantic_overlap(topic1, topic2)
                    topic_scores.append(distinctiveness)

            if topic_scores:
                distinctiveness_scores.append(float(np.mean(topic_scores)))

        return {
            'topic_distinctiveness': distinctiveness_scores,
            'average_distinctiveness': float(np.mean(distinctiveness_scores)) if distinctiveness_scores else 0.0
        }

    def _calculate_semantic_overlap(self, topic1: List[str], topic2: List[str]) -> float:
        """토픽 간 중복도 계산 - 개선된 버전"""
        emb1 = self._get_topic_embedding(topic1[:10], use_weights=True)
        emb2 = self._get_topic_embedding(topic2[:10], use_weights=True)

        semantic_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))

        # 키워드 중복 가중치 조정
        top5_common = set(topic1[:5]) & set(topic2[:5])
        top10_common = set(topic1[5:10]) & set(topic2[5:10])

        # 중복 단어에 대한 페널티 감소
        word_overlap = (len(top5_common) * 1.2 + len(top10_common)) / 8.0

        # 가중치 조정
        overlap_score = (
            0.4 * semantic_sim.item() +  # 의미적 유사도 비중 증가
            0.6 * word_overlap          # 단어 중복 비중 감소
        )

        # 중복도가 높은 경우의 페널티 완화
        if overlap_score > 0.4:
            overlap_score *= 1.4

        distinctiveness = 1.0 - overlap_score
        return min(1.0, max(0.0, distinctiveness))

    def evaluate(self, topics: List[List[str]], docs: List[str]) -> Dict[str, Any]:
        """토픽 모델 전체 평가 수행"""
        try:
            coherence_result = self.evaluate_topic_coherence(topics)
            distinctiveness_result = self.evaluate_topic_distinctiveness(topics)

            raw_scores = {
                'coherence': coherence_result['average_coherence'],
                'distinctiveness': distinctiveness_result['average_distinctiveness']
            }

            weights = {
                'coherence': 0.6,      # 토픽의 의미적 품질
                'distinctiveness': 0.4  # 토픽 간 구별성
            }

            overall_score = (
                raw_scores['coherence'] * weights['coherence'] +
                raw_scores['distinctiveness'] * weights['distinctiveness']
            )

            return {
                'raw_scores': raw_scores,
                'weights': weights,
                'weighted_scores': {
                    'coherence': raw_scores['coherence'] * weights['coherence'],
                    'distinctiveness': raw_scores['distinctiveness'] * weights['distinctiveness']
                },
                'overall_score': overall_score,
                'topic_coherence': coherence_result['topic_coherence'],
                'topic_distinctiveness': distinctiveness_result['topic_distinctiveness']
            }

        except Exception as e:
            print(f"[ERROR] 평가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {}

    def set_model_embeddings(self, embeddings: Dict[str, torch.Tensor], embedding_dim: int):
        """Set model embeddings for evaluation."""
        self.model_embeddings = embeddings
        self.embedding_dim = embedding_dim


# 코드 정리한 평가지표

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Union
from scipy.stats import entropy, jensenshannon
import traceback
import math
import os

class TopicModelNeuralEvaluator:
    def __init__(self, 
                 model_embeddings: Dict[str, torch.Tensor], 
                 embedding_dim: int = 768, 
                 device: str = 'cpu'):
        """Initialize neural topic model evaluator"""
        self.model_embeddings = model_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.missing_embeddings = set()
        
    def _get_embedding(self, keyword: str) -> torch.Tensor:
        """키워드 임베딩 반환"""
        try:
            if keyword in self.model_embeddings:
                emb = self.model_embeddings[keyword]
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                if emb.dtype != torch.float32:
                    emb = emb.to(torch.float32)
                return emb.to(self.device)
            
            if self.embedding_dim is None:
                raise ValueError("임베딩 차원이 설정되지 않았습니다.")
            
            self.missing_embeddings.add(keyword)
            default_emb = torch.randn(self.embedding_dim, device=self.device, dtype=torch.float32)
            default_emb = F.normalize(default_emb, p=2, dim=0)
            self.model_embeddings[keyword] = default_emb
            return default_emb
            
        except Exception as e:
            print(f"[ERROR] 임베딩 변환 중 오류 발생: {str(e)}")
            return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)

    def _calculate_term_weights(self, keywords: List[str]) -> torch.Tensor:
        """계층적 키워드 가중치 계산"""
        position_weights = torch.tensor([
            1.2, 1.1, 1.0, 0.9, 0.85,    # 최상위 키워드
            0.7, 0.6, 0.5, 0.4, 0.35,    # 중위 키워드
            0.3, 0.25, 0.2, 0.15, 0.1    # 하위 키워드
        ], device=self.device)
        
        keywords = keywords[:15]
        weights = position_weights[:len(keywords)]
        return F.softmax(weights, dim=0)

    def _get_topic_embedding(self, topic_keywords: List[str], use_weights: bool = True) -> torch.Tensor:
        """토픽 임베딩 계산"""
        if not topic_keywords:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        keyword_embeddings = []
        valid_keywords = []
        
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            if not torch.any(torch.isnan(emb)):
                keyword_embeddings.append(emb)
                valid_keywords.append(keyword)
                
        if not keyword_embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        if use_weights:
            weights = self._calculate_term_weights(valid_keywords)
        else:
            weights = torch.ones(len(keyword_embeddings), device=self.device) / len(keyword_embeddings)
            
        embeddings_tensor = torch.stack(keyword_embeddings)
        weighted_sum = torch.sum(embeddings_tensor * weights.unsqueeze(1), dim=0)
        
        return F.normalize(weighted_sum, p=2, dim=0)

    def evaluate_topic_coherence(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 일관성 평가"""
        coherence_scores = []
        position_weights = self._calculate_term_weights(['dummy'] * 10)
        
        for topic_keywords in topics:
            if not topic_keywords:
                coherence_scores.append(0.0)
                continue
                
            top_keywords = topic_keywords[:10]
            vectors = []
            
            for keyword in top_keywords:
                vec = self._get_embedding(keyword)
                if not torch.any(torch.isnan(vec)):
                    vectors.append(vec)
                    
            if len(vectors) < 2:
                coherence_scores.append(0.0)
                continue
                
            vectors = torch.stack(vectors)
            
            # 핵심 키워드 응집도
            core_coherence = 0.0
            core_pairs = 0
            for i in range(min(5, len(vectors))):
                for j in range(i+1, min(5, len(vectors))):
                    sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                    weight = position_weights[i] * position_weights[j] * 2.0
                    core_coherence += sim.item() * weight
                    core_pairs += weight
            
            core_coherence = core_coherence / core_pairs if core_pairs > 0 else 0.0
            
            # 전체 키워드 응집도
            full_coherence = 0.0
            total_weight = 0.0
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    weight = position_weights[i] * position_weights[j]
                    sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                    full_coherence += sim.item() * weight
                    total_weight += weight
            
            full_coherence = full_coherence / total_weight if total_weight > 0 else 0.0
            
            # 주제 명확성
            core_vectors = vectors[:5]
            center = torch.mean(core_vectors, dim=0)
            topic_clarity = torch.mean(F.cosine_similarity(core_vectors, center.unsqueeze(0)))
            
            # 최종 점수
            raw_score = (
                0.5 * core_coherence +  # 핵심 키워드 응집도
                0.3 * full_coherence +  # 전체 키워드 응집도
                0.2 * topic_clarity     # 주제 명확성
            )
            
            transformed_score = 1 / (1 + math.exp(-5 * (raw_score - 0.5)))
            final_score = 0.05 + 0.9 * transformed_score
            
            if final_score > 0.7:
                final_score = min(0.95, final_score * 1.2)
            elif final_score < 0.3:
                final_score = max(0.05, final_score * 0.8)
            
            coherence_scores.append(float(final_score))
        
        return {
            'topic_coherence': coherence_scores,
            'average_coherence': float(np.mean(coherence_scores)) if coherence_scores else 0.0
        }

    def _detect_randomness(self, keywords: List[str]) -> float:
        """토픽 키워드의 무작위성 탐지"""
        if len(keywords) < 2:
            return 1.0
            
        try:
            similarity_matrix = torch.zeros((len(keywords), len(keywords)), device=self.device)
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    emb1 = self._get_embedding(keywords[i])
                    emb2 = self._get_embedding(keywords[j])
                    similarity_matrix[i, j] = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    similarity_matrix[j, i] = similarity_matrix[i, j]
            
            similarity_matrix = similarity_matrix.cpu().numpy()
            entropy_scores = []
            
            for i in range(len(keywords)):
                row = similarity_matrix[i]
                probs = np.abs(row) / np.sum(np.abs(row))
                entropy_val = entropy(probs + 1e-10)
                entropy_scores.append(entropy_val)
            
            avg_entropy = np.mean(entropy_scores)
            normalized_randomness = min(1.0, avg_entropy / np.log(len(keywords)))
            
            return normalized_randomness
            
        except Exception as e:
            print(f"[WARNING] 무작위성 탐지 중 오류 발생: {str(e)}")
            return 0.5

    def evaluate_topic_distinctiveness(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 간 구별성 평가"""
        if not topics:
            return {'topic_distinctiveness': [], 'average_distinctiveness': 0.0}
            
        distinctiveness_scores = []
        
        for i, topic1 in enumerate(topics):
            # 현재 토픽의 무작위성 검사
            randomness_score = self._detect_randomness(topic1)
            
            # 다른 토픽들과의 구별성 계산
            topic_scores = []
            for j, topic2 in enumerate(topics):
                if i != j:
                    base_distinctiveness = self._calculate_semantic_overlap(topic1, topic2)
                    penalized_distinctiveness = base_distinctiveness * (1.0 - randomness_score * 0.5)
                    topic_scores.append(penalized_distinctiveness)
            
            if topic_scores:
                distinctiveness_scores.append(float(np.mean(topic_scores)))
        
        return {
            'topic_distinctiveness': distinctiveness_scores,
            'average_distinctiveness': float(np.mean(distinctiveness_scores)) if distinctiveness_scores else 0.0
        }

    def _calculate_semantic_overlap(self, topic1: List[str], topic2: List[str]) -> float:
        """토픽 간 중복도 계산"""
        randomness1 = self._detect_randomness(topic1)
        randomness2 = self._detect_randomness(topic2)
                
        randomness_score = max(randomness1, randomness2)
        penalty = torch.sigmoid(torch.tensor(20 * (randomness_score - 0.8))) * 4 + 0.2
        
        emb1 = self._get_topic_embedding(topic1[:10], use_weights=True)
        emb2 = self._get_topic_embedding(topic2[:10], use_weights=True)
        
        semantic_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        
        top5_common = set(topic1[:5]) & set(topic2[:5])
        top10_common = set(topic1[5:10]) & set(topic2[5:10])
        
        word_overlap = (len(top5_common) * 1.5 + len(top10_common)) / 7.5
        
        overlap_score = (
            0.3 * semantic_sim.item() +
            0.7 * word_overlap
        )
        
        if overlap_score > 0.4:
            overlap_score *= 1.8
        
        overlap_score = overlap_score * penalty
        distinctiveness = 1.0 - overlap_score
        
        return min(1.0, max(0.0, distinctiveness))

    def evaluate_topic_distribution(self, topic_assignments: List[int], num_topics: int) -> Dict[str, Any]:
        """토픽 분포 균형성 평가"""
        try:
            topic_counts = Counter(topic_assignments)
            distribution = np.zeros(num_topics)
            
            for topic_id in range(num_topics):
                distribution[topic_id] = topic_counts.get(topic_id, 0)
            
            if np.sum(distribution) == 0:
                return self._get_default_evaluation_result()
            
            distribution = distribution / len(topic_assignments)
            
            # 지니 계수
            gini = self._calculate_gini_coefficient(distribution)
            normalized_gini = 1 - gini
            
            # JS 거리
            uniform_dist = np.ones(num_topics) / num_topics
            js_divergence = jensenshannon(distribution, uniform_dist)
            normalized_js = 1 - float(js_divergence) if not np.isnan(js_divergence) else 0.0
            
            # 토픽 누락 패널티
            missing_topics = sum(1 for count in distribution if count == 0)
            missing_ratio = missing_topics / num_topics
            missing_penalty = (1 - missing_ratio) ** 1.5
            
            # 의미적 균형성
            semantic_balance = self._calculate_semantic_balance(distribution)
            
            # 최종 점수
            base_score = (
                0.3 * normalized_gini +
                0.3 * normalized_js +
                0.4 * semantic_balance
            ) * missing_penalty
            
            distribution_score = 0.3 + 0.6 * base_score
            
            return {
                'topic_distribution': distribution.tolist(),
                'distribution_score': distribution_score,
                'gini_score': normalized_gini,
                'js_score': normalized_js,
                'missing_penalty': missing_penalty,
                'semantic_balance': semantic_balance
            }
            
        except Exception as e:
            print(f"[ERROR] 토픽 분포 평가 중 오류 발생: {str(e)}")
            return self._get_default_evaluation_result()

    def _calculate_gini_coefficient(self, distribution: np.ndarray) -> float:
        """지니 계수 계산"""
        try:
            sorted_dist = np.sort(distribution)
            n = len(sorted_dist)
            if n == 0 or np.sum(sorted_dist) == 0:
                return 1.0
            
            cumsum = np.cumsum(sorted_dist)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            return np.clip(gini, 0.0, 1.0)
            
        except Exception as e:
            print(f"[ERROR] 지니 계수 계산 중 오류 발생: {str(e)}")
            return 1.0

    def _calculate_semantic_balance(self, distribution: np.ndarray) -> float:
        """의미적 균형성 계산"""
        try:
            mean = np.mean(distribution)
            std = np.std(distribution)
            if std == 0:
                return 0.0
            kurtosis = np.mean(((distribution - mean) / std) ** 4)
            
            entropy_score = entropy(distribution + 1e-10)
            max_entropy = np.log(len(distribution))
            normalized_entropy = entropy_score / max_entropy if max_entropy > 0 else 0
            
            uniformity = 1 - np.std(distribution) / np.mean(distribution) if np.mean(distribution) > 0 else 0
            
            semantic_balance = (
                0.4 * normalized_entropy + 
                0.4 * (1 / (1 + kurtosis)) +
                0.2 * uniformity
            )
            
            return semantic_balance
            
        except Exception as e:
            print(f"[ERROR] 의미적 균형성 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _get_default_evaluation_result(self) -> Dict[str, Any]:
        """기본 평가 결과 반환"""
        return {
            'topic_distribution': [],
            'distribution_score': 0.0,
            'gini_score': 0.0,
            'js_score': 0.0,
            'semantic_balance': 0.0
        }

    def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        """토픽 모델 전체 평가 수행"""
        try:
            coherence_result = self.evaluate_topic_coherence(topics)
            distinctiveness_result = self.evaluate_topic_distinctiveness(topics)
            distribution_result = self.evaluate_topic_distribution(topic_assignments, len(topics))
            
            raw_scores = {
                'coherence': coherence_result['average_coherence'],
                'distinctiveness': distinctiveness_result['average_distinctiveness'],
                'distribution': distribution_result['distribution_score']
            }
            
            weights = {
                'coherence': 0.5,        # 토픽의 의미적 품질
                'distinctiveness': 0.2,   # 토픽 간 구별성
                'distribution': 0.3       # 분포 균형성
            }
            
            overall_score = (
                raw_scores['coherence'] * weights['coherence'] +
                raw_scores['distinctiveness'] * weights['distinctiveness'] +
                raw_scores['distribution'] * weights['distribution']
            )
            
            return {
                'raw_scores': raw_scores,
                'weights': weights,
                'weighted_scores': {
                    'coherence': raw_scores['coherence'] * weights['coherence'],
                    'distinctiveness': raw_scores['distinctiveness'] * weights['distinctiveness'],
                    'distribution': raw_scores['distribution'] * weights['distribution']
                },
                'overall_score': overall_score,
                'topic_coherence': coherence_result['topic_coherence'],
                'topic_distinctiveness': distinctiveness_result['topic_distinctiveness'],
                'topic_distribution': distribution_result
            }
            
        except Exception as e:
            print(f"[ERROR] 평가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {}


# 계층까지 고려한 매우 복잡 평가

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Dict, Any, List, Union
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import traceback
import os

class TopicModelNeuralEvaluator:
    def __init__(self, 
                 model_embeddings: Dict[str, torch.Tensor], 
                 embedding_dim: int = 768, 
                 device: str = 'cpu',
                 cache_paths: Dict[str, str] = None):
        """Initialize neural topic model evaluator"""
        self.model_embeddings = model_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.missing_embeddings = set()
        self.cached_similarities = {}
        
        self.cache_paths = cache_paths or {
            'base_cache_dir': os.path.join('cache', 'default'),
            'embeddings_cache': os.path.join('cache', 'default', 'embeddings.pkl')
        }
        
        os.makedirs(self.cache_paths['base_cache_dir'], exist_ok=True)

    def set_model_embeddings(self, embeddings: Union[Dict[str, torch.Tensor], torch.Tensor], embedding_dim: int = None, vocab: Dict[str, int] = None) -> None:
        """모델 임베딩 설정"""
        try:
            self.model_embeddings = embeddings
            if embedding_dim is not None:
                self.embedding_dim = embedding_dim
            if vocab is not None:
                self.vocab = vocab
            self.missing_embeddings.clear()
            self.cached_similarities.clear()
        except Exception as e:
            print(f"[ERROR] 모델 임베딩 설정 중 오류 발생: {str(e)}")

    def _get_embedding(self, keyword: str) -> torch.Tensor:
        """키워드 임베딩 반환"""
        try:
            # 딕셔너리 타입 체크
            if isinstance(self.model_embeddings, dict):
                if keyword in self.model_embeddings:
                    emb = self.model_embeddings[keyword]
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb)
                    if emb.dtype != torch.float32:
                        emb = emb.to(torch.float32)
                    return emb.to(self.device)
            # 텐서 타입 체크
            elif isinstance(self.model_embeddings, torch.Tensor):
                if hasattr(self, 'vocab') and keyword in self.vocab:
                    idx = self.vocab[keyword]
                    emb = self.model_embeddings[idx]
                    if emb.dtype != torch.float32:
                        emb = emb.to(torch.float32)
                    return emb.to(self.device)
            
            if self.embedding_dim is None:
                raise ValueError("임베딩 차원이 설정되지 않았습니다.")
            
            self.missing_embeddings.add(keyword)
            default_emb = torch.randn(self.embedding_dim, device=self.device, dtype=torch.float32)
            default_emb = F.normalize(default_emb, p=2, dim=0)
            return default_emb
            
        except Exception as e:
            print(f"[ERROR] 임베딩 변환 중 오류 발생: {str(e)}")
            return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)

    def _get_topic_embedding(self, topic_keywords: List[str], use_weights: bool = True) -> torch.Tensor:
        """토픽 임베딩 계산"""
        if not topic_keywords:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        keyword_embeddings = []
        weights = []
        
        # 키워드별 임베딩 및 가중치 계산
        for idx, keyword in enumerate(topic_keywords):
            emb = self._get_embedding(keyword)
            if not torch.any(torch.isnan(emb)):
                keyword_embeddings.append(emb)
                
                if use_weights:
                    # 위치 기반 가중치
                    position_weight = 1.0 / (1 + idx * 0.2)
                    weights.append(position_weight)
        
        if not keyword_embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # 임베딩 결합
        embeddings_tensor = torch.stack(keyword_embeddings)
        
        if use_weights:
            weights_tensor = torch.tensor(weights, device=self.device)
            weights_tensor = F.softmax(weights_tensor, dim=0)
            weighted_sum = torch.sum(embeddings_tensor * weights_tensor.unsqueeze(1), dim=0)
            return F.normalize(weighted_sum, p=2, dim=0)
        else:
            mean_embedding = torch.mean(embeddings_tensor, dim=0)
            return F.normalize(mean_embedding, p=2, dim=0)

    def evaluate_topic_coherence(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 일관성 평가"""
        coherence_scores = []
        
        for topic in topics:
            if not topic:
                coherence_scores.append(0.0)
                continue
                
            # 핵심 키워드(상위 5개)에 대해 더 엄격한 평가
            core_coherence = self._calculate_core_coherence(topic[:5])
            
            # 전체 키워드 응집도는 보조적으로만 사용
            full_coherence = self._calculate_full_coherence(topic)
            
            # 비중 조정
            final_score = 0.7 * core_coherence + 0.3 * full_coherence
            
            # 무작위성 검사 강화
            if self._detect_randomness(topic) > 0.4:
                final_score *= 0.5
                
            coherence_scores.append(float(final_score))
        
        return {
            'topic_coherence': coherence_scores,
            'average_coherence': float(np.mean(coherence_scores)) if coherence_scores else 0.0
        }

    def _calculate_core_coherence(self, keywords: List[str]) -> float:
        """핵심 키워드에 대한 일관성 계산"""
        if len(keywords) < 2:
            return 0.0
            
        window_size = 5
        contextual_scores = []
        
        # 슬라이딩 윈도우로 지역적 문맥 분석
        for i in range(len(keywords) - window_size + 1):
            window = keywords[i:i + window_size]
            window_similarities = []
            
            # 윈도우 내 단어들의 관계 분석
            for j in range(len(window)):
                for k in range(j + 1, len(window)):
                    w1, w2 = window[j], window[k]
                    cache_key = tuple(sorted([w1, w2]))
                    
                    if cache_key in self.cached_similarities:
                        sim = self.cached_similarities[cache_key]
                    else:
                        emb1 = self._get_embedding(w1)
                        emb2 = self._get_embedding(w2)
                        sim = F.cosine_similarity(
                            emb1.unsqueeze(0),
                            emb2.unsqueeze(0)
                        ).item()
                        self.cached_similarities[cache_key] = sim
                    
                    # 거리 기반 가중치
                    weight = 1.0 / (1 + abs(j - k))
                    window_similarities.append(sim * weight)
            
            if window_similarities:
                contextual_scores.append(np.mean(window_similarities))
        
        return np.mean(contextual_scores) if contextual_scores else 0.0
    
    def _calculate_full_coherence(self, keywords: List[str]) -> float:
        """전체 키워드에 대한 일관성 계산"""
        if len(keywords) < 3:
            return 0.0
            
        # 상위 키워드들의 중심성 계산
        top_k = min(5, len(keywords))
        top_keywords = keywords[:top_k]
        
        top_embeddings = [self._get_embedding(w) for w in top_keywords]
        center = torch.mean(torch.stack(top_embeddings), dim=0)
        
        hierarchical_scores = []
        
        # 각 키워드의 계층적 관계 평가
        for i, keyword in enumerate(keywords):
            emb = self._get_embedding(keyword)
            
            # 상위 키워드는 중심과의 관계
            if i < top_k:
                sim = F.cosine_similarity(emb.unsqueeze(0), center.unsqueeze(0))
                weight = 1.0
            # 하위 키워드는 상위 키워드들과의 관계
            else:
                sims = []
                for top_emb in top_embeddings:
                    sim = F.cosine_similarity(emb.unsqueeze(0), top_emb.unsqueeze(0))
                    sims.append(sim.item())
                sim = torch.tensor(np.mean(sims), device=self.device)
                weight = 0.8
            
            hierarchical_scores.append(sim.item() * weight)
        
        return np.mean(hierarchical_scores) if hierarchical_scores else 0.0

    def _detect_randomness(self, keywords: List[str]) -> float:
        """무작위성 검사"""
        if len(keywords) < 2:
            return 0.0
            
        # 키워드 간 유사도 계산
        similarity_matrix = torch.zeros((len(keywords), len(keywords)), device=self.device)
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                emb1 = self._get_embedding(keywords[i])
                emb2 = self._get_embedding(keywords[j])
                similarity_matrix[i, j] = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                similarity_matrix[j, i] = similarity_matrix[i, j]
        
        # 유사도 행렬의 엔트로피 계산
        entropy_matrix = -similarity_matrix * torch.log(similarity_matrix)
        entropy_matrix[torch.isnan(entropy_matrix)] = 0.0
        entropy_matrix[torch.isinf(entropy_matrix)] = 0.0
        entropy_matrix = torch.sum(entropy_matrix, dim=0) / len(keywords)
        
        # 엔트로피의 평균 계산
        average_entropy = torch.mean(entropy_matrix).item()
        
        # 엔트로피 기반 무작위성 평가
        if average_entropy < 0.1:
            return 0.0
        elif average_entropy < 0.2:
            return 0.2
        elif average_entropy < 0.3:
            return 0.4
        elif average_entropy < 0.4:
            return 0.6
        else:
            return 1.0

    def evaluate_topic_distinctiveness(self, topics: List[List[str]]) -> Dict[str, float]:
        """토픽 구별성 평가"""
        distinctiveness_scores = []
        
        for i, topic1 in enumerate(topics):
            if not topic1:
                distinctiveness_scores.append(0.0)
                continue
            
            topic_distincts = []
            for j, topic2 in enumerate(topics):
                if i != j and topic2:
                    distinctiveness = self.calculate_semantic_overlap(topic1, topic2)
                    topic_distincts.append(distinctiveness)
            
            if topic_distincts:
                # 평균과 최소값을 모두 고려
                avg_distinctiveness = np.mean(topic_distincts)
                min_distinctiveness = np.min(topic_distincts)
                
                # 최종 점수 - 최소값에 더 큰 가중치
                final_score = (0.4 * avg_distinctiveness + 0.6 * min_distinctiveness)
                distinctiveness_scores.append(final_score)
            else:
                distinctiveness_scores.append(0.0)
        
        return {
            'topic_distinctiveness': distinctiveness_scores,
            'average_distinctiveness': np.mean(distinctiveness_scores) if distinctiveness_scores else 0.0
        }

    def calculate_semantic_overlap(self, topic1, topic2):
        # 토픽 임베딩 계산
        emb1 = self._get_topic_embedding(topic1[:10])
        emb2 = self._get_topic_embedding(topic2[:10])
        
        # 기본 유사도 계산
        semantic_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
        # 단어 중복 가중치 강화
        word_overlap = len(set(topic1[:10]) & set(topic2[:10])) / 10
        
        # 수정된 중복도 계산
        overlap_score = 0.4 * semantic_sim + 0.6 * word_overlap
        
        # 임계값 기반 패널티 (더 엄격하게)
        if overlap_score > 0.3:  # 임계값 하향
            overlap_score *= 1.5  # 패널티 상향
            
        return 1.0 - overlap_score  # 구별성 점수

    def evaluate_topic_distribution(self, topic_assignments: List[int], num_topics: int) -> Dict[str, Any]:
        """토픽 분포 평가"""
        try:
            # 토픽 할당 분포 계산
            topic_counts = Counter(topic_assignments)
            distribution = np.zeros(num_topics)
            
            for topic_id in range(num_topics):
                distribution[topic_id] = topic_counts.get(topic_id, 0)
            
            if np.sum(distribution) == 0:
                return self._get_default_evaluation_result()
            
            # 1. 분포 정규화
            distribution = distribution / len(topic_assignments)
            
            # 2. 지니 계수 계산
            gini = self._calculate_gini_coefficient(distribution)
            normalized_gini = 1 - gini
            
            # 3. JS 거리 계산
            uniform_dist = np.ones(num_topics) / num_topics
            js_divergence = jensenshannon(distribution, uniform_dist)
            normalized_js = 1 - float(js_divergence) if not np.isnan(js_divergence) else 0.0
            
            # 4. 토픽 누락 패널티
            missing_topics = sum(1 for count in distribution if count == 0)
            missing_ratio = missing_topics / num_topics
            missing_penalty = (1 - missing_ratio) ** 1.5
            
            # 5. 최종 분포 점수 계산
            distribution_score = (
                0.4 * normalized_gini +    # 균형성
                0.4 * normalized_js +      # 균일성
                0.2 * missing_penalty      # 누락 패널티
            )
            
            return {
                'distribution_score': distribution_score,
                'gini_score': normalized_gini,
                'js_score': normalized_js,
                'missing_penalty': missing_penalty
            }
            
        except Exception as e:
            print(f"[ERROR] 토픽 분포 평가 중 오류 발생: {str(e)}")
            return self._get_default_evaluation_result()

    def _calculate_gini_coefficient(self, distribution: np.ndarray) -> float:
        """지니 계수 계산"""
        try:
            sorted_dist = np.sort(distribution)
            n = len(sorted_dist)
            if n == 0 or np.sum(sorted_dist) == 0:
                return 1.0
            
            cumsum = np.cumsum(sorted_dist)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            return np.clip(gini, 0.0, 1.0)
            
        except Exception as e:
            print(f"[ERROR] 지니 계수 계산 중 오류: {str(e)}")
            return 1.0

    def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        """토픽 모델 종합 평가"""
        try:
            if not self._validate_inputs(topics, docs, topic_assignments):
                return {
                    'coherence': {'topic_coherence': [], 'average_coherence': 0.0},
                    'distinctiveness': {'topic_distinctiveness': [], 'average_distinctiveness': 0.0},
                    'distribution': self._get_default_evaluation_result(),
                    'overall_score': 0.0
                }
            
            # 1. 각 측면 평가
            coherence_result = self.evaluate_topic_coherence(topics)
            distinctiveness_result = self.evaluate_topic_distinctiveness(topics)
            distribution_result = self.evaluate_topic_distribution(topic_assignments, len(topics))
            
            # 2. Unknown 토픽 패널티
            unknown_ratio = sum(1 for t in topics if self._is_unknown_topic(t)) / len(topics)
            unknown_penalty = 1.0 - unknown_ratio
            
            # 3. 점수 범위 정규화 (0.0 ~ 1.0)
            coh_score = coherence_result['average_coherence']  # 이미 0~1
            dist_score = distinctiveness_result['average_distinctiveness']  # 이미 0~1
            distr_score = distribution_result['distribution_score']  # 이미 0~1
            
            # 4. 최종 점수 계산
            overall_score = (
                0.4 * coh_score +      # 일관성
                0.4 * dist_score +     # 구별성
                0.2 * distr_score      # 분포
            ) * unknown_penalty
            
            return {
                'coherence': coherence_result,
                'distinctiveness': distinctiveness_result,
                'distribution': distribution_result,
                'overall_score': overall_score,
                'raw_scores': {
                    'coherence': coh_score,
                    'distinctiveness': dist_score,
                    'distribution': distr_score,
                    'unknown_penalty': unknown_penalty
                }
            }
            
        except Exception as e:
            print(f"[ERROR] 토픽 모델 평가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {
                'coherence': {'topic_coherence': [], 'average_coherence': 0.0},
                'distinctiveness': {'topic_distinctiveness': [], 'average_distinctiveness': 0.0},
                'distribution': self._get_default_evaluation_result(),
                'overall_score': 0.0
            }

    def _validate_inputs(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> bool:
        """입력 데이터 검증"""
        try:
            if not topics or not all(isinstance(topic, list) for topic in topics):
                print("[ERROR] 잘못된 토픽 형식")
                return False
            
            if not docs:
                print("[ERROR] 문서가 비어있음")
                return False
            
            if len(docs) != len(topic_assignments):
                print("[ERROR] 문서 수와 토픽 할당 수가 일치하지 않음")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 입력 검증 중 오류 발생: {str(e)}")
            return False

    def _get_default_evaluation_result(self) -> Dict[str, Any]:
        """기본 평가 결과"""
        return {
            'distribution_score': 0.0,
            'gini_score': 0.0,
            'js_score': 0.0,
            'missing_penalty': 0.0
        }

    def _is_unknown_topic(self, keywords: List[str]) -> bool:
        """Unknown 토픽 탐지"""
        unknown_patterns = {
            'unknown', 'etc', 'miscellaneous', 
            'other', 'undefined', 'unclassified',
            'misc', 'general', 'various', 'mixed',
            'uncategorized', 'remaining', 'rest',
            'unlabeled', 'na', 'none'
        }
        return any(p in k.lower() for k in keywords[:5] for p in unknown_patterns)



# 통계적 평가지표

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
                        
                        score = np.log((d_w1w2 + 1) / (d_w1 + 1))
                        topic_coherence.append(score)
                
                if topic_coherence:
                    coherence_scores.append(np.mean(topic_coherence))
            
            if coherence_scores:
                mean_score = np.mean(coherence_scores)
                return 1 / (1 + np.exp(-mean_score))
            return 0.0
            
        except Exception as e:
            print(f"[WARNING] Error calculating coherence: {str(e)}")
            return 0.0

    # 2. 구별성(KL-Divergence) 평가
    def _calculate_kld(self, topics: List[List[str]], topic_assignments: List[int], **kwargs) -> float:
        """KL Divergence 계산"""
        valid_assignments = [t for t in topic_assignments if t >= 0]
        if not valid_assignments:
            return 0.0
            
        topic_dist = np.bincount(valid_assignments, minlength=len(topics))
        if np.sum(topic_dist) == 0:
            return 0.0
            
        topic_dist = topic_dist / np.sum(topic_dist)
        uniform_dist = np.ones_like(topic_dist) / len(topic_dist)
        
        return entropy(topic_dist, uniform_dist)

    # 3. 다양성(Diversity) 평가
    def _calculate_diversity(self, topics: List[List[str]]) -> float:
        """토픽 다양성 계산"""
        try:
            if not topics:
                return 0.0
            
            # 상위 10개 키워드만 사용
            n = 10
            
            unique_words = set()
            total_words = 0
            
            for topic in topics:
                if not topic:
                    continue
                words = topic[:n]
                unique_words.update(words)
                total_words += len(words)
            
            if total_words == 0:
                return 0.0
                
            return len(unique_words) / total_words
            
        except Exception as e:
            print(f"[WARNING] Diversity 계산 중 오류: {str(e)}")
            return 0.0

    # 4. 분포 점수(Distribution Score) 평가
    def evaluate_topic_distribution_quality(self, docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        """문서 집합의 분포 평가"""
        try:
            topic_distribution = self.get_topic_distribution(topic_assignments)
            
            if topic_distribution is None:
                topic_distribution = np.array([])
            
            if self.natural_distribution is None:
                num_topics = len(topic_distribution)
                self.natural_distribution = np.ones(num_topics) / num_topics
            
            distribution_similarity = self.evaluate_distributions(
                self.natural_distribution, 
                topic_distribution,
                metrics=['jensen_shannon_divergence', 'wasserstein_distance', 'kullback_leibler_divergence']
            )
            
            if distribution_similarity is None:
                distribution_similarity = {}
            
            cluster_alignment = self.evaluate_cluster_alignment(
                self.natural_clusters if self.natural_clusters is not None else topic_assignments,
                topic_assignments,
                metrics=['adjusted_mutual_info', 'adjusted_rand_index', 'normalized_mutual_info']
            )
            
            similarity_score = np.mean(list(distribution_similarity.values()))
            alignment_score = np.mean(list(cluster_alignment.values()))
            final_score = 0.7 * similarity_score + 0.3 * alignment_score
            
            return {
                'score': final_score,
                'metrics': {
                    'distribution_similarity': distribution_similarity,
                    'cluster_alignment': cluster_alignment
                }
            }
        
        except Exception as e:
            return self.handle_evaluation_error(e, "토픽 분포 품질 평가")


    def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        """토픽 모델 종합 평가"""
        try:
            self._validate_inputs(topics, docs, topic_assignments)
            
            # 각 지표의 원점수 계산 (기존 메서드 사용)
            coherence_score = self._calculate_coherence(topics)
            distinctiveness_score = self._calculate_kld(topics, topic_assignments)
            distribution_result = self.evaluate_topic_distribution_quality(docs, topic_assignments)
            
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
                'coherence': 0.4,        # 토픽 품질
                'distinctiveness': 0.4,   # 토픽 간 구별성
                'distribution': 0.2       # 분포 균형성
            }
            
            # 원점수 
            raw_scores = {
                'coherence': coherence_score,
                'distinctiveness': distinctiveness_score,
                'distribution': distribution_result['score']
            }
            
            # 가중치 적용된 개별 점수
            weighted_scores = {
                'coherence': raw_scores['coherence'] * weights['coherence'],
                'distinctiveness': raw_scores['distinctiveness'] * weights['distinctiveness'],
                'distribution': raw_scores['distribution'] * weights['distribution']
            }
            
            # 최종 종합 점수
            overall_score = sum(weighted_scores.values())
            
            return {
                'coherence': coherence_result,
                'distinctiveness': distinctiveness_result,
                'distribution': distribution_result,
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
                'distribution': self._get_default_evaluation_result(),
                'raw_scores': {'coherence': 0.0, 'distinctiveness': 0.0, 'distribution': 0.0},
                'weighted_scores': {'coherence': 0.0, 'distinctiveness': 0.0, 'distribution': 0.0},
                'weights': {'coherence': 0.4, 'distinctiveness': 0.4, 'distribution': 0.2},
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

    def get_topic_distribution(self, topic_assignments: List[int]) -> np.ndarray:
        """토픽 할당 분포 계산"""
        try:
            topic_distribution = Counter(topic_assignments)
            return self.normalize_distribution(dict(topic_distribution))
        except Exception as e:
            print(f"[ERROR] 토픽 분포 생성 중 오류: {str(e)}")
            return np.array([])
        
    # 데이터 검증 및 정규화 메서드
    def _validate_inputs(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> None:
        """Validate input data"""
        if not topics or not all(isinstance(topic, list) for topic in topics):
            raise ValueError("Invalid topic format")
        
        if not docs:
            raise ValueError("Documents are empty")
        
        if len(docs) != len(topic_assignments):
            raise ValueError("Number of documents does not match topic assignments")

    def normalize_distribution(self, distribution):
        """분포 데이터 정규화"""
        pass

    def evaluate_distributions(self, natural_distribution: np.ndarray, topic_distribution: np.ndarray, metrics: List[str]) -> Dict[str, float]:
        """분포 유사도 평가"""
        results = {}
        try:
            # 상위 10개 토픽만 고려
            natural_distribution = natural_distribution[:10] if len(natural_distribution) > 10 else natural_distribution
            topic_distribution = topic_distribution[:10] if len(topic_distribution) > 10 else topic_distribution
            
            if len(natural_distribution) == 0 or len(topic_distribution) == 0:
                return {metric: 0.0 for metric in metrics}

            max_len = max(len(natural_distribution), len(topic_distribution))
            nat_dist = np.pad(natural_distribution, (0, max_len - len(natural_distribution)))
            top_dist = np.pad(topic_distribution, (0, max_len - len(topic_distribution)))

            for metric in metrics:
                if metric == 'jensen_shannon_divergence':
                    results[metric] = jensenshannon(nat_dist, top_dist)
                elif metric == 'wasserstein_distance':
                    results[metric] = wasserstein_distance(nat_dist, top_dist)
                elif metric == 'kullback_leibler_divergence':
                    results[metric] = entropy(nat_dist, top_dist)

        except Exception as e:
            print(f"[ERROR] 분포 유사도 계산 중 오류: {str(e)}")
            traceback.print_exc()
            results = {metric: 0.0 for metric in metrics}

        return results

    def evaluate_cluster_alignment(self, natural_clusters, topic_assignments, metrics: List[str]) -> Dict[str, float]:
        """클러스터 일치도 평가"""
        results = {}
        try:
            for metric in metrics:
                if metric == 'adjusted_mutual_info':
                    results[metric] = adjusted_mutual_info_score(natural_clusters, topic_assignments)
                elif metric == 'adjusted_rand_index':
                    results[metric] = adjusted_rand_score(natural_clusters, topic_assignments)
                elif metric == 'normalized_mutual_info':
                    results[metric] = normalized_mutual_info_score(natural_clusters, topic_assignments)
        except Exception as e:
            print(f"[ERROR] 토러스터 일치도 계산 중 오류: {str(e)}")
            traceback.print_exc()
            results = {metric: 0.0 for metric in metrics}

        return results

    def handle_evaluation_error(self, error: Exception, method_name: str) -> Dict[str, float]:
        """Standardized error handling"""
        print(f"[ERROR] Error during {method_name}: {str(error)}")
        traceback.print_exc()
        return {
            'score': 0.0,
            'metrics': {
                'distribution_similarity': 0.0,
                'cluster_alignment': 0.0,
                'coherence': 0.0,
                'distinctiveness': 0.0,
                'diversity': 0.0,
                'imbalance_penalty': 0.0,
                'total': 0.0
            }
        }
        
        
    # def _calculate_coherence(self, topics: List[List[str]]) -> float:
    #     """토픽 일관성 계산 (NPMI 기반)"""
    #     try:
    #         if not topics:
    #             return 0.0
            
    #         word_doc_freq = self.model_stats.get('word_doc_freq', {})
    #         co_doc_freq = self.model_stats.get('co_doc_freq', {})
    #         total_docs = self.total_documents or len(word_doc_freq)
        
#         if total_docs == 0:
#             return 0.0
        
    #         coherence_scores = []
    #         for topic_words in topics:
    #             if len(topic_words) < 2:
    #                 continue
                
    #             topic_coherence = []
    #             for i in range(1, len(topic_words)):
    #                 for j in range(0, i):
    #                     word2, word1 = topic_words[i], topic_words[j]
                    
    #                     # 각 단어의 문서 빈도 계산
    #                     p_w1 = word_doc_freq.get(word1, 0) / total_docs
    #                     p_w2 = word_doc_freq.get(word2, 0) / total_docs
                    
    #                     if p_w1 == 0 or p_w2 == 0:
    #                         continue
                        
    #                     # 공출현 확률 계산
    #                     pair = (word1, word2) if word1 < word2 else (word2, word1)
    #                     p_w1w2 = (co_doc_freq.get(pair, 0) + 1e-10) / total_docs  # smoothing
                    
    #                     # NPMI 계산
    #                     pmi = np.log(p_w1w2 / (p_w1 * p_w2))
    #                     npmi = pmi / (-np.log(p_w1w2))  # [-1, 1] 범위
                    
    #                     # [0, 1] 범위로 정규화
    #                     normalized_npmi = (npmi + 1) / 2
    #                     topic_coherence.append(normalized_npmi)
            
    #             if topic_coherence:
    #                 coherence_scores.append(np.mean(topic_coherence))
        
    #         # 최종 coherence score 계산
    #         if coherence_scores:
    #             return float(np.mean(coherence_scores))
    #         return 0.0
        
    #     except Exception as e:
    #         print(f"[WARNING] NPMI Coherence 계산 중 오류: {str(e)}")
    #         return 0.0

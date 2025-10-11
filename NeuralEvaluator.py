# 다면적 특성을 종합적으로 고려 2차 수정

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Union, Tuple
from scipy.stats import entropy, zscore
from scipy.spatial.distance import pdist, squareform
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

    def _analyze_semantic_structure(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """토픽 내 의미적 구조 분석"""
        # 페어와이즈 유사도 계산
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # 유사도 분포 분석
        sim_upper = torch.triu(sim_matrix, diagonal=1)
        similarities = sim_upper[sim_upper != 0]
        
        if len(similarities) == 0:
            return {
                'mean_sim': 0.0,
                'sim_std': 0.0,
                'connectivity': 0.0
            }

        # 기본 통계량
        mean_sim = torch.mean(similarities).item()
        sim_std = torch.std(similarities).item()
        
        # 연결성 분석 (높은 유사도 연결의 비율)
        high_sim_ratio = torch.sum(similarities > 0.5) / len(similarities)
        connectivity = float(high_sim_ratio)

        return {
            'mean_sim': mean_sim,
            'sim_std': sim_std,
            'connectivity': connectivity
        }

    def _detect_semantic_outliers(self, embeddings: torch.Tensor) -> float:
        """의미적 이상치 탐지"""
        # 중심점으로부터의 거리 계산
        centroid = torch.mean(embeddings, dim=0)
        distances = 1 - F.cosine_similarity(embeddings, centroid.unsqueeze(0))
        
        # Modified Z-score 방식으로 이상치 탐지
        median_dist = torch.median(distances)
        mad = torch.median(torch.abs(distances - median_dist))
        modified_zscores = 0.6745 * (distances - median_dist) / (mad + 1e-10)
        
        # 이상치 비율 계산
        outlier_ratio = torch.sum(modified_zscores > 3.0) / len(distances)
        
        return float(outlier_ratio)

    def _calculate_coherence_score(self, topic_keywords: List[str]) -> float:
        """토픽 일관성 점수 계산 (OLD method: PageRank + Hierarchical Similarity)"""
        if len(topic_keywords) < 2:
            return 0.0

        # 임베딩 준비
        embeddings = []
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            if not torch.any(torch.isnan(emb)):
                embeddings.append(emb)

        if len(embeddings) < 2:
            return 0.0

        embeddings_tensor = torch.stack(embeddings)
        n_keywords = len(embeddings)

        # 1. 의미 그래프 구축 및 PageRank로 중요도 계산
        from networkx import Graph, pagerank
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(embeddings_tensor.cpu().numpy())
        graph = Graph()

        # 그래프 구축 (임계값 기반 엣지 생성)
        for i in range(n_keywords):
            for j in range(i + 1, n_keywords):
                if similarities[i, j] > 0.3:
                    graph.add_edge(i, j, weight=float(similarities[i, j]))

        # PageRank로 키워드 중요도 계산
        try:
            importance_scores = pagerank(graph)
        except:
            # 그래프가 비어있거나 연결되지 않은 경우 균등 분포
            importance_scores = {i: 1.0/n_keywords for i in range(n_keywords)}

        # 2. 계층적 유사도 계산 (직접적 + 간접적 유사도)
        direct_sim = torch.tensor(similarities, device=self.device)

        # 간접적 유사도 (2차 관계까지)
        indirect_sim = torch.matmul(direct_sim, direct_sim) / n_keywords

        # 직접적 유사도와 간접적 유사도 결합 (0.7:0.3 비율)
        hierarchical_sim = 0.7 * direct_sim + 0.3 * indirect_sim

        # 3. 중요도 가중치 행렬 생성
        importance_matrix = np.array([[importance_scores.get(i, 1.0/n_keywords) *
                                      importance_scores.get(j, 1.0/n_keywords)
                                      for j in range(n_keywords)]
                                     for i in range(n_keywords)])

        # 4. 가중치가 적용된 일관성 점수 계산
        weighted_similarities = hierarchical_sim.cpu().numpy()
        coherence_score = (weighted_similarities * importance_matrix).sum() / \
                         (importance_matrix.sum() + 1e-10)  # 0으로 나누는 것 방지

        # 0~1 범위로 제한
        return float(min(1.0, max(0.0, coherence_score)))

    def _calculate_distinctiveness_score(self, topic1: List[str], topic2: List[str]) -> float:
        """토픽 간 구별성 점수 계산"""
        # 토픽 임베딩 계산
        emb1 = self._get_topic_embedding(topic1[:10])
        emb2 = self._get_topic_embedding(topic2[:10])
        
        # 의미적 유사도
        semantic_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        
        # 키워드 중복 분석
        common_words = set(topic1) & set(topic2)
        word_overlap = len(common_words) / min(len(topic1), len(topic2))
        
        # 구별성 점수 계산
        distinctiveness = 1.0 - (
            0.7 * semantic_sim.item() +  # 의미적 유사도에 더 큰 가중치
            0.3 * word_overlap  
        )
        
        return float(distinctiveness)

    def _get_topic_embedding(self, topic_keywords: List[str]) -> torch.Tensor:
        """토픽 임베딩 계산"""
        embeddings = []
        for keyword in topic_keywords:
            emb = self._get_embedding(keyword)
            if not torch.any(torch.isnan(emb)):
                embeddings.append(emb)
                
        if not embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        embeddings_tensor = torch.stack(embeddings)
        mean_embedding = torch.mean(embeddings_tensor, dim=0)
        
        return F.normalize(mean_embedding, p=2, dim=0)

    def evaluate_topic_coherence(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 일관성 평가"""
        coherence_scores = []
        
        for topic_keywords in topics:
            coherence = self._calculate_coherence_score(topic_keywords)
            coherence_scores.append(coherence)

        return {
            'topic_coherence': coherence_scores,
            'average_coherence': float(np.mean(coherence_scores)) if coherence_scores else 0.0
        }

    def evaluate_topic_distinctiveness(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 구별성 평가"""
        if not topics:
            return {'topic_distinctiveness': [], 'average_distinctiveness': 0.0}

        distinctiveness_scores = []

        for i, topic1 in enumerate(topics):
            topic_scores = []
            for j, topic2 in enumerate(topics):
                if i != j:
                    distinctiveness = self._calculate_distinctiveness_score(topic1, topic2)
                    topic_scores.append(distinctiveness)

            if topic_scores:
                distinctiveness_scores.append(float(np.mean(topic_scores)))

        return {
            'topic_distinctiveness': distinctiveness_scores,
            'average_distinctiveness': float(np.mean(distinctiveness_scores)) if distinctiveness_scores else 0.0
        }

    def evaluate_topic_diversity(self, topics: List[List[str]]) -> Dict[str, Any]:
        """토픽 다양성 평가 (Topic Diversity)

        TD = unique words / total words
        토픽 간 중복되지 않는 고유 단어의 비율을 측정
        """
        if not topics:
            return {'diversity': 0.0}

        all_words = set()
        total_words = 0

        for topic_keywords in topics:
            all_words.update(topic_keywords)
            total_words += len(topic_keywords)

        diversity = len(all_words) / total_words if total_words > 0 else 0.0

        return {
            'diversity': float(diversity),
            'unique_words': len(all_words),
            'total_words': total_words
        }

    def evaluate(self, topics: List[List[str]], docs: List[str]) -> Dict[str, Any]:
        """통합 평가 수행"""
        try:
            coherence_result = self.evaluate_topic_coherence(topics)
            distinctiveness_result = self.evaluate_topic_distinctiveness(topics)
            diversity_result = self.evaluate_topic_diversity(topics)

            raw_scores = {
                'coherence': coherence_result['average_coherence'],
                'distinctiveness': distinctiveness_result['average_distinctiveness'],
                'diversity': diversity_result['diversity']
            }

            weights = {
                'coherence': 0.5,       # 일관성
                'distinctiveness': 0.3,  # 구별성
                'diversity': 0.2         # 다양성
            }

            weighted_scores = {
                'coherence': raw_scores['coherence'] * weights['coherence'],
                'distinctiveness': raw_scores['distinctiveness'] * weights['distinctiveness'],
                'diversity': raw_scores['diversity'] * weights['diversity']
            }

            overall_score = (weighted_scores['coherence'] +
                           weighted_scores['distinctiveness'] +
                           weighted_scores['diversity'])

            return {
                'raw_scores': raw_scores,
                'weighted_scores': weighted_scores,
                'overall_score': overall_score,
                'topic_coherence': coherence_result['topic_coherence'],
                'topic_distinctiveness': distinctiveness_result['topic_distinctiveness'],
                'diversity_info': {
                    'diversity': diversity_result['diversity'],
                    'unique_words': diversity_result['unique_words'],
                    'total_words': diversity_result['total_words']
                }
            }

        except Exception as e:
            print(f"[ERROR] 평가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {}

    def set_model_embeddings(self, embeddings: Dict[str, torch.Tensor], embedding_dim: int):
        """Set model embeddings for evaluation"""
        self.model_embeddings = embeddings
        self.embedding_dim = embedding_dim

    def _detect_incoherent_structure(self, embeddings_tensor: torch.Tensor) -> float:
        """토픽 내부 유사도 패턴 검사"""
        # 페어와이즈 유사도 계산
        similarities = F.cosine_similarity(
            embeddings_tensor.unsqueeze(1),
            embeddings_tensor.unsqueeze(0),
            dim=2
        )
        
        # 상위 키워드들 간의 유사도 평균
        top_k_sim = torch.mean(similarities[:5, :5])
        
        # 전체 유사도의 표준편차
        total_std = torch.std(similarities)
        
        # 비일관성 점수 (높을수록 구조가 부자연스러움)
        incoherence_score = (1 - top_k_sim) * (1 + total_std)
        
        return float(incoherence_score)
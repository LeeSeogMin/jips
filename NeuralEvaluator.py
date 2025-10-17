import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import os
from scipy.stats import entropy
from networkx import Graph, pagerank
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

class TopicModelNeuralEvaluator:
    def __init__(self, device=None, data_dir='data'):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir

        # Load pre-computed embeddings and topics
        for data_type in ['distinct', 'similar', 'more_similar']:
            embeddings_path = os.path.join(self.data_dir, f'embeddings_{data_type}.pkl')
            topics_path = os.path.join(self.data_dir, f'topics_{data_type}.pkl')

            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Required embeddings file not found: {embeddings_path}")
            if not os.path.exists(topics_path):
                raise FileNotFoundError(f"Required topics file not found: {topics_path}")

            with open(embeddings_path, 'rb') as f:
                setattr(self, f'embeddings_{data_type}', torch.tensor(pickle.load(f)).to(self.device))
            with open(topics_path, 'rb') as f:
                setattr(self, f'topics_{data_type}', pickle.load(f))

    def _get_keyword_embeddings(self, keyword: str) -> List[torch.Tensor]:
        """특정 키워드의 임베딩 벡터들을 모든 데이터셋에서 검색"""
        embeddings = []
        for data_type in ['distinct', 'similar', 'more_similar']:
            indices = [i for i, words in enumerate(getattr(self, f'topics_{data_type}'))
                      if keyword in words]
            if indices:
                embeddings.append(getattr(self, f'embeddings_{data_type}')[indices].mean(dim=0))
        return embeddings

    def _get_topic_embeddings(self, topic_keywords: List[str]) -> List[torch.Tensor]:
        """토픽의 키워드들에 대한 임베딩 벡터들을 모든 데이터셋에서 검색"""
        embeddings = []
        for data_type in ['distinct', 'similar', 'more_similar']:
            indices = [i for i, words in enumerate(getattr(self, f'topics_{data_type}'))
                      if any(keyword in words for keyword in topic_keywords)]
            if indices:
                embeddings.append(getattr(self, f'embeddings_{data_type}')[indices].mean(dim=0))
        return embeddings

    def _calculate_similarity_metrics(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """임베딩 벡터들 간의 유사도 메트릭스 계산"""
        similarities = torch.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        mask = torch.triu(torch.ones_like(similarities), diagonal=1)
        return {'similarities': similarities, 'mask': mask}
    
    def _build_semantic_graph(self, embeddings: torch.Tensor, keywords: List[str]) -> Tuple[Graph, Dict[int, float]]:
        """의미적 관계 그래프 구축 및 중요도 계산"""
        similarities = cosine_similarity(embeddings.cpu())
        graph = Graph()
        
        # 그래프 구축
        n_keywords = len(keywords)
        for i in range(n_keywords):
            for j in range(i + 1, n_keywords):
                if similarities[i, j] > 0.3:  # 임계값 기반 엣지 생성
                    graph.add_edge(i, j, weight=similarities[i, j])
        
        # PageRank로 키워드 중요도 계산
        importance_scores = pagerank(graph)
        return graph, importance_scores

    def _calculate_hierarchical_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """계층적 의미 관계를 고려한 유사도 계산"""
        n_keywords = embeddings.size(0)
        
        # 직접적인 유사도
        direct_sim = torch.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        
        # 간접적인 유사도 (2차 관계까지)
        indirect_sim = torch.matmul(direct_sim, direct_sim) / n_keywords
        
        # 직접적 유사도와 간접적 유사도 결합
        hierarchical_sim = 0.7 * direct_sim + 0.3 * indirect_sim
        return hierarchical_sim

    def _calculate_topic_entropy(self, similarities: torch.Tensor) -> float:
        """토픽 내 의미적 다양성 계산"""
        prob_dist = F.softmax(similarities.mean(dim=0), dim=0)
        return float(entropy(prob_dist.cpu().numpy()))

    def _evaluate_semantic_coherence(self, topics: List[List[str]]) -> Dict[str, Any]:
        """향상된 의미적 일관성 평가"""
        topic_coherence_scores = []
        avg_coherence = 0.0  # 초기값 설정
        
        # 모든 토픽의 키워드 수를 동일하게 맞추기
        min_keywords = min(len(topic) for topic in topics)
        topics = [topic[:min_keywords] for topic in topics]
        
        for topic_idx, topic_keywords in enumerate(topics):
            keyword_embeddings = []
            for keyword in topic_keywords:
                embeddings = self._get_keyword_embeddings(keyword)
                if embeddings:
                    keyword_embeddings.extend(embeddings)
            
            if keyword_embeddings:
                keyword_embeddings = torch.stack(keyword_embeddings)
                
                # 의미 그래프 구축 및 중요도 계산
                graph, importance_scores = self._build_semantic_graph(
                    keyword_embeddings, topic_keywords)
                
                # 계층적 유사도 계산
                hierarchical_similarities = self._calculate_hierarchical_similarity(
                    keyword_embeddings)
                
                # 엔트로피 기반 다양성 계산
                topic_entropy = self._calculate_topic_entropy(hierarchical_similarities)
                
                # 가중치가 적용된 일관성 점수 계산
                weighted_similarities = hierarchical_similarities.cpu().numpy()
                
                # importance_matrix 생성
                n_keywords = len(topic_keywords)
                importance_scores = {i: importance_scores.get(i, 1.0/(i+1)) 
                                for i in range(n_keywords)}
                
                importance_matrix = np.array([[importance_scores[i] * importance_scores[j]
                                            for j in range(n_keywords)]
                                            for i in range(n_keywords)])
                
                # 행렬 크기 확인 및 조정
                n = len(weighted_similarities)
                m = len(importance_matrix)
                if n != m:
                    min_size = min(n, m)
                    weighted_similarities = weighted_similarities[:min_size, :min_size]
                    importance_matrix = importance_matrix[:min_size, :min_size]
                
                coherence_score = (weighted_similarities * importance_matrix).sum() / \
                                (importance_matrix.sum() + 1e-10)  # 0으로 나누는 것 방지
                
                topic_coherence_scores.append(float(coherence_score))
            else:
                topic_coherence_scores.append(0.0)
        
        # 점수가 있는 경우에만 평균 계산
        if topic_coherence_scores:
            avg_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)
        
        return {
            'topic_coherence': topic_coherence_scores,
            'average_coherence': avg_coherence,
        }

    def _evaluate_topic_distinctiveness(self, topics: List[List[str]]) -> Dict[str, float]:
        """토픽 간 구별성 평가"""
        n_topics = len(topics)
        distinctiveness_matrix = torch.zeros((n_topics, n_topics), device=self.device)
        pair_scores = {}
        
        # 각 토픽의 평균 임베딩 계산
        topic_embeddings = []
        for topic_keywords in topics:
            embeddings = self._get_topic_embeddings(topic_keywords)
            if embeddings:
                topic_embeddings.append(torch.stack(embeddings).mean(dim=0))
            else:
                topic_embeddings.append(torch.zeros(self.embeddings_distinct.shape[1], device=self.device))
        
        topic_embeddings = torch.stack(topic_embeddings)
        metrics = self._calculate_similarity_metrics(topic_embeddings)
        
        # 유사도를 구별성 점수로 변환 (1 - similarity) / 2
        distinctiveness = (1 - metrics['similarities']) / 2
        
        # 상삼각행렬 값들만 추출하여 pair_scores 구성
        mask = metrics['mask'].bool()
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                pair_scores[f'pair_{i}_{j}'] = distinctiveness[i,j].item()
        
        return {
            'distinctiveness_matrix': distinctiveness.cpu().numpy().tolist(),
            'average_distinctiveness': float(np.mean(list(pair_scores.values()))),
            'topic_pairs': pair_scores
        }

    def _evaluate_semantic_diversity(self, topics: List[List[str]]) -> Dict[str, float]:
        """토픽들의 의미적 다양성을 계산"""
        if len(topics) < 2:
            return {'diversity_scores': [], 'average_diversity': 0.0}
        
        # 토픽 임베딩 계산
        topic_embeddings = []
        for topic_keywords in topics:
            embeddings = self._get_topic_embeddings(topic_keywords)
            if embeddings:
                topic_embeddings.append(torch.stack(embeddings).mean(dim=0))
            else:
                topic_embeddings.append(torch.zeros(self.embeddings_distinct.shape[1], device=self.device))
        
        topic_embeddings = torch.stack(topic_embeddings)
        metrics = self._calculate_similarity_metrics(topic_embeddings)
        
        # 다양성 점수 계산 (1 - similarity) / 2
        mask = metrics['mask'].bool()
        diversity_scores = (1 - metrics['similarities'][mask]) / 2
        
        # 토픽 쌍별 다양성 점수 저장
        pair_scores = {}
        n_topics = len(topics)
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                pair_scores[f'pair_{i}_{j}'] = float((1 - metrics['similarities'][i,j]) / 2)
        
        return {
            'diversity_scores': diversity_scores.cpu().numpy().tolist(),
            'average_diversity': float(diversity_scores.mean().item()),
            'topic_pairs': pair_scores
        }

    def calculate_distribution_diversity(self, topic_assignments: List[int]) -> Dict[str, float]:
        """토픽 할당의 분포적 다양성을 계산"""
        valid_assignments = [t for t in topic_assignments if t >= 0]
        
        if not valid_assignments:
            return {
                'distribution_diversity': 0.0,
                'topic_proportions': {}
            }
        
        N = max(valid_assignments) + 1  # 토픽의 총 개수
        valid_count = len(valid_assignments)
        
        # 토픽별 비율 계산
        topic_counts = np.bincount(valid_assignments, minlength=N)
        P_T = topic_counts / valid_count
        
        # 엔트로피 계산
        P_T += 1e-12  # 0 확률 방지
        H_T = -np.sum(P_T * np.log(P_T))
        H_max = np.log(N)
        distribution_diversity = H_T / H_max
        
        # 토픽별 비율 저장
        topic_proportions = {
            f'topic_{i}': float(P_T[i])
            for i in range(N)
        }
        
        return {
            'distribution_diversity': float(distribution_diversity),
            'topic_proportions': topic_proportions
        }

    def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        """토픽 모델의 전체적인 품질을 평가"""
        # 기존 평가 메트릭 계산
        coherence_scores = self._evaluate_semantic_coherence(topics)
        distinctiveness_scores = self._evaluate_topic_distinctiveness(topics)
        
        # 추가된 다양성 메트릭 계산
        semantic_diversity_scores = self._evaluate_semantic_diversity(topics)
        distribution_diversity_scores = self.calculate_distribution_diversity(topic_assignments)
        
        # 전체 다양성 점수 계산 (의미적 다양성과 분포적 다양성의 평균)
        overall_diversity = (
            semantic_diversity_scores['average_diversity'] +
            distribution_diversity_scores['distribution_diversity']
        ) / 2
        
        return {
            'Coherence': coherence_scores['average_coherence'],
            'Distinctiveness': distinctiveness_scores['average_distinctiveness'],
            'Diversity': overall_diversity,
            'detailed_scores': {
                'Coherence': coherence_scores,
                'Distinctiveness': distinctiveness_scores,
                'semantic_diversity': semantic_diversity_scores,
                'distribution_diversity': distribution_diversity_scores
            }
        }
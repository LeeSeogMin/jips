import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import os
from scipy.stats import entropy
from networkx import Graph, pagerank
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch import nn
import logging

class TopicModelNeuralEvaluator:
    def __init__(self, device=None, data_dir='../data'):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        
        # Initialize sentence transformer model for dynamic embeddings
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load pre-computed embeddings and topics (optional for newsgroup validation)
        self.use_dynamic_embeddings = False
        for data_type in ['distinct', 'similar', 'more_similar']:
            embeddings_path = os.path.join(self.data_dir, f'embeddings_{data_type}.pkl')
            topics_path = os.path.join(self.data_dir, f'topics_{data_type}.pkl')

            if os.path.exists(embeddings_path) and os.path.exists(topics_path):
                with open(embeddings_path, 'rb') as f:
                    setattr(self, f'embeddings_{data_type}', torch.tensor(pickle.load(f)).to(self.device))
                with open(topics_path, 'rb') as f:
                    setattr(self, f'topics_{data_type}', pickle.load(f))
            else:
                # Enable dynamic embeddings for newsgroup validation
                self.use_dynamic_embeddings = True

    def _get_keyword_embeddings(self, keyword: str) -> List[torch.Tensor]:
        """특정 키워드의 임베딩 벡터들을 모든 데이터셋에서 검색"""
        embeddings = []
        
        # Use dynamic embeddings for newsgroup validation
        if self.use_dynamic_embeddings:
            # Generate embedding on-the-fly for newsgroup validation
            embedding = self.model.encode([keyword], convert_to_tensor=True)
            return [embedding[0]]
        
        for data_type in ['distinct', 'similar', 'more_similar']:
            if hasattr(self, f'topics_{data_type}'):
                indices = [i for i, words in enumerate(getattr(self, f'topics_{data_type}'))
                          if keyword in words]
                if indices and hasattr(self, f'embeddings_{data_type}'):
                    embeddings.append(getattr(self, f'embeddings_{data_type}')[indices].mean(dim=0))
        return embeddings

    def _get_topic_embeddings(self, topic_keywords: List[str]) -> List[torch.Tensor]:
        """토픽의 키워드들에 대한 임베딩 벡터들을 모든 데이터셋에서 검색"""
        embeddings = []
        
        # Use dynamic embeddings for newsgroup validation
        if self.use_dynamic_embeddings:
            # Generate embeddings on-the-fly for newsgroup validation
            topic_embeddings = self.model.encode(topic_keywords, convert_to_tensor=True)
            return [topic_embeddings]
        
        for data_type in ['distinct', 'similar', 'more_similar']:
            if hasattr(self, f'topics_{data_type}'):
                indices = [i for i, words in enumerate(getattr(self, f'topics_{data_type}'))
                          if any(keyword in words for keyword in topic_keywords)]
                if indices and hasattr(self, f'embeddings_{data_type}'):
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

class TopicSemanticIntegration:
    def __init__(self, base_evaluator: TopicModelNeuralEvaluator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_evaluator = base_evaluator

        # 임베딩 차원을 base_evaluator의 임베딩 차원과 맞춤
        embedding_dim = self.base_evaluator.embeddings_distinct.shape[1]

        self.integration_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim//2, embedding_dim//4),
            nn.ReLU(),
            nn.Linear(embedding_dim//4, 1),
            nn.Sigmoid()
        ).to(self.device)

    def compute_integration_score(self, topics: List[List[str]], docs: List[str], topic_count: int) -> float:
        try:
            topic_scores = []
            topic_embeddings = self._get_topic_embeddings(topics)

            for topic_idx, topic_words in enumerate(topics):
                # 의미적 일관성 계산
                coherence_score = self._compute_semantic_coherence(topic_words, topic_embeddings[topic_idx])

                # 의미적 구별성 계산
                distinctiveness_score = self._compute_semantic_distinctiveness(
                    topic_embeddings[topic_idx],
                    topic_embeddings,
                    topic_idx
                )

                # 임베딩 공간에서의 분포 품질 계산
                distribution_score = self._compute_embedding_distribution(topic_embeddings[topic_idx])

                # 최종 토픽 점수 계산
                topic_score = (
                    coherence_score * 0.4 +
                    distinctiveness_score * 0.4 +
                    distribution_score * 0.2
                )
                topic_scores.append(topic_score)

            # 전체 토픽의 의미적 구조 평가
            semantic_structure_score = self._evaluate_semantic_structure(topic_embeddings)

            # 최종 통합 점수 계산
            final_score = np.mean(topic_scores) * semantic_structure_score

            return final_score

        except Exception as e:
            logging.error(f"Error in semantic integration score computation: {e}")
            return 0.0

    def _get_topic_embeddings(self, topics: List[List[str]]) -> torch.Tensor:
        """토픽별 임베딩 계산"""
        topic_embeddings = []

        for topic_words in topics:
            # base_evaluator의 _get_topic_embeddings 메서드 활용
            embeddings = self.base_evaluator._get_topic_embeddings(topic_words)
            if embeddings:
                topic_emb = torch.stack(embeddings).mean(dim=0)
            else:
                topic_emb = torch.zeros(self.base_evaluator.embeddings_distinct.shape[1],
                                      device=self.device)
            topic_embeddings.append(topic_emb)

        return torch.stack(topic_embeddings)

    def _compute_semantic_coherence(self, topic_words: List[str], topic_emb: torch.Tensor) -> float:
        """임베딩 기반 의미적 일관성 계산"""
        try:
            word_embeddings = []
            for word in topic_words:
                word_embs = self.base_evaluator._get_keyword_embeddings(word)
                if word_embs:
                    word_embeddings.append(torch.stack(word_embs).mean(dim=0))

            if not word_embeddings:
                return 0.0

            word_embeddings = torch.stack(word_embeddings)

            similarities = torch.cosine_similarity(
                word_embeddings,
                topic_emb.unsqueeze(0),
                dim=1
            )

            weights = torch.linspace(1.0, 0.5, len(similarities), device=self.device)
            weighted_similarity = (similarities * weights).mean()

            return weighted_similarity.item()

        except Exception as e:
            logging.error(f"Error in semantic coherence computation: {e}")
            return 0.0

    def _compute_semantic_distinctiveness(self, topic_emb: torch.Tensor,
                                       all_topic_embs: torch.Tensor,
                                       current_idx: int) -> float:
        """임베딩 기반 의미적 구별성 계산"""
        try:
            similarities = torch.cosine_similarity(
                topic_emb.unsqueeze(0),
                all_topic_embs,
                dim=1
            )

            similarities = torch.cat([
                similarities[:current_idx],
                similarities[current_idx+1:]
            ])

            distinctiveness = 1 - similarities

            distinctiveness_sorted, _ = torch.sort(distinctiveness)
            weights = torch.linspace(0.5, 1.0, len(distinctiveness_sorted), device=self.device)
            weighted_distinctiveness = (distinctiveness_sorted * weights).mean()

            return weighted_distinctiveness.item()

        except Exception as e:
            logging.error(f"Error in semantic distinctiveness computation: {e}")
            return 1.0

    def _compute_embedding_distribution(self, topic_emb: torch.Tensor) -> float:
        """임베딩 공간에서의 분포 품질 계산"""
        try:
            norm = torch.norm(topic_emb)
            if norm == 0:
                return 0.0

            normalized_emb = topic_emb / norm

            uniform_vector = torch.ones_like(topic_emb) / torch.sqrt(torch.tensor(topic_emb.shape[0]))
            distribution_quality = torch.cosine_similarity(
                normalized_emb.unsqueeze(0),
                uniform_vector.unsqueeze(0)
            )

            return distribution_quality.item()

        except Exception as e:
            logging.error(f"Error in embedding distribution computation: {e}")
            return 0.0

    def _evaluate_semantic_structure(self, topic_embeddings: torch.Tensor) -> float:
        """전체 토픽의 의미적 구조 평가"""
        try:
            similarities = torch.cosine_similarity(
                topic_embeddings.unsqueeze(1),
                topic_embeddings.unsqueeze(0),
            dim=2
        )
        
            mask = torch.ones_like(similarities) - torch.eye(len(topic_embeddings), device=self.device)
            similarities = similarities * mask

            similarity_mean = similarities[mask.bool()].mean()
            similarity_std = similarities[mask.bool()].std()

            structure_score = (1 - similarity_mean) * (1 - similarity_std)

            return structure_score.item()

        except Exception as e:
            logging.error(f"Error in semantic structure evaluation: {e}")
            return 1.0

class EnhancedTopicModelNeuralEvaluator(TopicModelNeuralEvaluator):
    def __init__(self, device=None):
        super().__init__(device=device)
        self.semantic_integration_evaluator = TopicSemanticIntegration(base_evaluator=self, device=device)

    def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
        # 기존 평가 결과 가져오기
        base_results = super().evaluate(topics, docs, topic_assignments)

        # 의미적 통합 평가 수행
        integration_score = self.semantic_integration_evaluator.compute_integration_score(
            topics=topics,
            docs=docs,
            topic_count=len(topics)
        )

        # 결과 통합
        enhanced_results = {
            **base_results,
            'Semantic Integration Score': integration_score,
            'Overall Score': (
                base_results['Coherence'] * 0.3 +
                base_results['Distinctiveness'] * 0.3 +
                base_results['Diversity'] * 0.2 +
                integration_score * 0.2
            )
        }

        return enhanced_results
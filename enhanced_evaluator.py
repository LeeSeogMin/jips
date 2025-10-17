import torch
import numpy as np
from typing import List, Dict, Any
from torch import nn
import logging
from NeuralEvaluator import TopicModelNeuralEvaluator

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


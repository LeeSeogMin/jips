"""
Base Topic Model class for evaluation framework
"""

from typing import Dict, Any, List
import torch

class BaseTopicModel:
    """Base class for topic models with evaluation capabilities"""

    def __init__(self, num_topics: int):
        self.num_topics = num_topics
        self.fitted = False
        self.stat_evaluator = None
        self.neural_evaluator = None

    def set_evaluators(self, embeddings: Dict[str, torch.Tensor]) -> None:
        """
        Initialize evaluators with word embeddings

        Args:
            embeddings: Dictionary mapping words to embedding tensors
        """
        try:
            from .StatEvaluator import TopicModelStatEvaluator
            from .NeuralEvaluator import TopicModelNeuralEvaluator

            self.stat_evaluator = TopicModelStatEvaluator()

            if embeddings:
                embedding_dim = next(iter(embeddings.values())).shape[0]
                self.neural_evaluator = TopicModelNeuralEvaluator(
                    model_embeddings=embeddings,
                    embedding_dim=embedding_dim,
                    device='cpu'
                )
        except Exception as e:
            print(f"[WARNING] Failed to initialize evaluators: {e}")

    def fit(self, tokenized_texts: List[List[str]], embeddings: Any) -> None:
        """
        Fit the topic model to data

        Args:
            tokenized_texts: List of tokenized documents
            embeddings: Document embeddings
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def get_topics(self) -> Dict[str, Any]:
        """
        Get extracted topics

        Returns:
            Dictionary containing topics and topic assignments
        """
        raise NotImplementedError("Subclasses must implement get_topics()")

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get model statistics for evaluation

        Returns:
            Dictionary containing model statistics
        """
        raise NotImplementedError("Subclasses must implement get_model_stats()")

    def evaluate(self, topics: List[List[str]], docs: List[str]) -> Dict[str, Any]:
        """
        Evaluate topics using both statistical and neural methods

        Args:
            topics: List of topic keywords
            docs: List of documents

        Returns:
            Dictionary containing evaluation results
        """
        results = {}

        if self.stat_evaluator:
            try:
                model_stats = self.get_model_stats()
                self.stat_evaluator.set_model_stats(
                    word_doc_freq=model_stats.get('word_doc_freq', {}),
                    co_doc_freq=model_stats.get('co_doc_freq', {}),
                    total_documents=model_stats.get('total_documents', len(docs)),
                    vocabulary_size=model_stats.get('vocabulary_size', 0),
                    topic_sizes=model_stats.get('topic_sizes', {})
                )

                stat_results = self.stat_evaluator.evaluate(
                    topics=topics,
                    docs=docs,
                    topic_assignments=self.get_topics().get('topic_assignments', [])
                )
                results['statistical'] = stat_results
            except Exception as e:
                print(f"[WARNING] Statistical evaluation failed: {e}")

        if self.neural_evaluator:
            try:
                neural_results = self.neural_evaluator.evaluate(
                    topics=topics,
                    docs=docs
                )
                results['neural'] = neural_results
            except Exception as e:
                print(f"[WARNING] Neural evaluation failed: {e}")

        return results

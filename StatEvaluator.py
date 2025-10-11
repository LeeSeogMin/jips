import numpy as np
import math
from typing import List, Dict, Any
from itertools import combinations
from collections import Counter

class TopicModelStatEvaluator:
   def __init__(self):
       """
       Statistical metrics-based topic model evaluator
       """
       pass

   def _normalize_scores(self, matrix: np.ndarray) -> np.ndarray:
       """
       점수를 0-1 범위로 정규화
       """
       min_val = np.min(matrix)
       max_val = np.max(matrix)
       if max_val - min_val > 0:
           return (matrix - min_val) / (max_val - min_val)
       return matrix

   def _evaluate_npmi(self, topics: List[List[str]], word_doc_freq: Dict[str, int],
                     co_doc_freq: Dict[tuple, int], total_documents: int) -> Dict[str, Any]:
       """
       토픽별 NPMI(Normalized Pointwise Mutual Information) 평가
       """
       epsilon = 1e-12
       npmi_results = {}
       
       for topic_idx, topic_words in enumerate(topics):
           npmi_scores = []
           pairs = combinations(topic_words, 2)
           
           for x_i, x_j in pairs:
               p_x_i = word_doc_freq.get(x_i, 0) / total_documents
               p_x_j = word_doc_freq.get(x_j, 0) / total_documents
               pair = tuple(sorted((x_i, x_j)))
               p_x_i_x_j = co_doc_freq.get(pair, 0) / total_documents
               
               if p_x_i_x_j == 0:
                   continue
                   
               numerator = math.log((p_x_i_x_j + epsilon) / (p_x_i * p_x_j + epsilon))
               denominator = -math.log(p_x_i_x_j + epsilon)
               npmi = numerator / denominator
               npmi_scores.append(npmi)
               
           if npmi_scores:
               npmi_results[f'topic_{topic_idx}'] = np.mean(npmi_scores)
           else:
               npmi_results[f'topic_{topic_idx}'] = 0.0

       # [-1, 1] 범위를 [0, 1]로 정규화
       normalized_scores = {k: (v + 1) / 2 for k, v in npmi_results.items()}
               
       return {
           'topic_scores': normalized_scores,
           'average_npmi': np.mean(list(normalized_scores.values()))
       }

   def _evaluate_cv_coherence(self, topics: List[List[str]], texts: List[List[str]], 
                            dictionary) -> Dict[str, Any]:
       """
       C_v Coherence 평가
       """
       coherence_model = CoherenceModel(
           topics=topics,
           texts=texts,
           dictionary=dictionary,
           coherence='c_v'
       )
       
       coherences = coherence_model.get_coherence_per_topic()
       coherences = np.array(coherences)
       
       # 정규화
       normalized_coherences = self._normalize_scores(coherences)
       
       coherence_results = {
           f'topic_{idx}': score for idx, score in enumerate(normalized_coherences)
       }
       
       return {
           'topic_scores': coherence_results,
           'average_coherence': float(np.mean(normalized_coherences))
       }

   def _calculate_coherence(self, topics: List[List[str]]) -> float:
       """Calculate topic coherence (evaluation 폴더 공식)"""
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

   def _calculate_jsd(self, topics: List[List[str]], topic_assignments: List[int]) -> float:
       """Jensen-Shannon Divergence 계산 (evaluation 폴더 공식)"""
       from scipy.spatial.distance import jensenshannon

       valid_assignments = [t for t in topic_assignments if t >= 0]
       if not valid_assignments:
           return 0.0

       topic_dist = np.bincount(valid_assignments, minlength=len(topics))
       if np.sum(topic_dist) == 0:
           return 0.0

       topic_dist = topic_dist / np.sum(topic_dist)
       uniform_dist = np.ones_like(topic_dist) / len(topic_dist)

       return jensenshannon(topic_dist, uniform_dist)

   def _calculate_diversity(self, topics: List[List[str]]) -> float:
       """토픽 다양성 계산 (Topic Diversity) - evaluation 폴더 공식

       TD = unique words / total words
       토픽 간 중복되지 않는 고유 단어의 비율을 측정
       """
       if not topics:
           return 0.0

       all_words = set()
       total_words = 0

       for topic_keywords in topics:
           all_words.update(topic_keywords)
           total_words += len(topic_keywords)

       diversity = len(all_words) / total_words if total_words > 0 else 0.0

       return diversity

   def set_model_stats(self, **kwargs):
       """모델 통계 정보 설정"""
       try:
           self.topic_sizes = kwargs.get('topic_sizes', {})
           self.vocabulary_size = kwargs.get('vocabulary_size', 0)
           self.total_documents = kwargs.get('total_documents', 0)
           self.model_stats = kwargs

       except Exception as e:
           print(f"[WARNING] 통계 정보 설정 중 오류 발생: {str(e)}")

   def _evaluate_kld(self, topics: List[List[str]], vocab_size: int = None) -> Dict[str, Any]:
       """
       KLD 값을 0-1 범위로 정규화
       """
       epsilon = 1e-12
       n_topics = len(topics)
       kld_matrix = np.zeros((n_topics, n_topics))
       pair_scores = {}
       
       # 전체 vocabulary 구성
       vocab = set()
       for topic in topics:
           vocab.update(topic)
       vocab = list(vocab)
       vocab_dict = {word: idx for idx, word in enumerate(vocab)}
       
       # 각 토픽의 단어 분포 계산
       topic_distributions = []
       for topic in topics:
           # 단어 빈도 계산
           word_freq = Counter(topic)
           
           # 분포 벡터 생성
           dist = np.zeros(len(vocab))
           for word, freq in word_freq.items():
               if word in vocab_dict:
                   dist[vocab_dict[word]] = freq
           
           # 확률 분포로 정규화
           total_count = np.sum(dist) + epsilon
           dist = dist / total_count
           topic_distributions.append(dist)
       
       # 토픽 쌍별 KLD 계산
       for i in range(n_topics):
           for j in range(n_topics):
               if i != j:
                   P = topic_distributions[i] + epsilon
                   Q = topic_distributions[j] + epsilon
                   
                   # KLD 계산: P(i) * log(P(i)/Q(i))
                   kld = np.sum(P * np.log(P / Q))
                   kld_matrix[i,j] = kld
   
       # KLD 값 정규화
       normalized_kld = self._normalize_scores(kld_matrix)
       
       # 정규화된 값으로 pair_scores 업데이트
       for i in range(n_topics):
           for j in range(n_topics):
               if i != j:
                   pair_scores[f'pair_{i}_{j}'] = float(normalized_kld[i,j])
       
       # 토픽별 평균 KLD 계산
       topic_avg_kld = {}
       for i in range(n_topics):
           topic_avg_kld[f'topic_{i}'] = float(np.mean(normalized_kld[i]))
       
       return {
           'kld_matrix': normalized_kld.tolist(),
           'average_kld': float(np.mean(list(pair_scores.values()))),
           'topic_pairs': pair_scores,
           'topic_avg_kld': topic_avg_kld
       }

   def _evaluate_jsd(self, topics: List[List[str]], vocab_size: int = None) -> Dict[str, Any]:
       """
       JSD 값을 0-1 범위로 정규화
       """
       epsilon = 1e-12
       n_topics = len(topics)
       jsd_matrix = np.zeros((n_topics, n_topics))
       pair_scores = {}
       
       # 전체 vocabulary 구성
       vocab = set()
       for topic in topics:
           vocab.update(topic)
       vocab = list(vocab)
       vocab_dict = {word: idx for idx, word in enumerate(vocab)}
       
       # 각 토픽의 단어 분포 계산
       topic_distributions = []
       for topic in topics:
           # 단어 빈도 계산
           word_freq = Counter(topic)
           
           # 분포 벡터 생성
           dist = np.zeros(len(vocab))
           for word, freq in word_freq.items():
               if word in vocab_dict:
                   dist[vocab_dict[word]] = freq
           
           # 확률 분포로 정규화
           total_count = np.sum(dist) + epsilon
           dist = dist / total_count
           topic_distributions.append(dist)
       
       # 토픽 쌍별 JSD 계산
       for i in range(n_topics):
           for j in range(i + 1, n_topics):
               P = topic_distributions[i]
               Q = topic_distributions[j]
               M = 0.5 * (P + Q)
               
               # JSD 계산: 0.5 * (KLD(P||M) + KLD(Q||M))
               P = P + epsilon
               Q = Q + epsilon
               M = M + epsilon
               
               jsd = 0.5 * (
                   np.sum(P * np.log(P / M)) +
                   np.sum(Q * np.log(Q / M))
               )
               
               jsd_matrix[i,j] = jsd
               jsd_matrix[j,i] = jsd  # JSD는 대칭적
   
       # JSD 값 정규화
       normalized_jsd = self._normalize_scores(jsd_matrix)
       
       # 정규화된 값으로 pair_scores 업데이트
       for i in range(n_topics):
           for j in range(i + 1, n_topics):
               pair_scores[f'pair_{i}_{j}'] = float(normalized_jsd[i,j])
       
       # 토픽별 평균 JSD 계산
       topic_avg_jsd = {}
       for i in range(n_topics):
           topic_avg_jsd[f'topic_{i}'] = float(np.mean(normalized_jsd[i]))
       
       return {
           'jsd_matrix': normalized_jsd.tolist(),
           'average_jsd': float(np.mean(list(pair_scores.values()))),
           'topic_pairs': pair_scores,
           'topic_avg_jsd': topic_avg_jsd
       }

   def _evaluate_irbo(self, topics: List[List[str]], p: float = 0.9) -> Dict[str, float]:
       """
       Inverted Rank-Biased Overlap (IRBO) 평가
       """
       K = len(topics)
       if K < 2:
           return {'irbo_score': 0.0}
           
       total_rbo = 0.0
       pair_scores = {}
       
       for i in range(K):
           for j in range(i + 1, K):
               # RBO 계산
               s, t = topics[i], topics[j]
               sl, tl = len(s), len(t)
               k = max(sl, tl)
               sum1 = 0.0
               
               for d in range(1, k + 1):
                   set_s = set(s[:d])
                   set_t = set(t[:d])
                   x_d = len(set_s.intersection(set_t))
                   weight = p ** (d - 1)
                   sum1 += (x_d / d) * weight
               
               rbo = (1 - p) * sum1
               total_rbo += rbo
               pair_scores[f'pair_{i}_{j}'] = float(rbo)
       
       avg_rbo = total_rbo / (K * (K - 1) / 2)
       irbo = 1 - avg_rbo
       
       return {
           'irbo_score': float(irbo),
           'pair_scores': pair_scores
       }

   def _compute_overall_score(self,
                            npmi_scores: Dict[str, Any],
                            coherence_scores: Dict[str, Any],
                            diversity_scores: Dict[str, float],
                            kld_scores: Dict[str, Any],
                            jsd_scores: Dict[str, Any],
                            irbo_scores: Dict[str, float]) -> float:
       """종합 평가 점수 계산"""
       weights = {
           'npmi': 0.2,
           'coherence': 0.2,
           'diversity': 0.15,
           'kld': 0.15,
           'jsd': 0.15,
           'irbo': 0.15
       }
       
       overall_score = (
           weights['npmi'] * npmi_scores['average_npmi'] +
           weights['coherence'] * coherence_scores['average_coherence'] +
           weights['diversity'] * diversity_scores['diversity_score'] +
           weights['kld'] * kld_scores['average_kld'] +
           weights['jsd'] * jsd_scores['average_jsd'] +
           weights['irbo'] * irbo_scores['irbo_score']
       )
       
       return overall_score

   def evaluate(self, topics: List[List[str]], docs: List[str], topic_assignments: List[int]) -> Dict[str, Any]:
       """토픽 모델 종합 평가 (evaluation 폴더 공식 사용)"""
       try:
           self._validate_inputs(topics, docs, topic_assignments)

           # 각 지표의 원점수 계산
           coherence_score = self._calculate_coherence(topics)
           distinctiveness_score = self._calculate_jsd(topics, topic_assignments)
           diversity_score = self._calculate_diversity(topics)

           # 결과를 새로운 형식으로 구성
           coherence_result = {
               'average_coherence': coherence_score,
               'topic_coherence': [coherence_score]
           }

           distinctiveness_result = {
               'average_distinctiveness': distinctiveness_score,
               'topic_distinctiveness': [distinctiveness_score]
           }

           diversity_result = {
               'diversity': diversity_score
           }

           # 고정 가중치 정의
           weights = {
               'coherence': 0.5,
               'distinctiveness': 0.3,
               'diversity': 0.2
           }

           # 원점수
           raw_scores = {
               'coherence': coherence_score,
               'distinctiveness': distinctiveness_score,
               'diversity': diversity_score
           }

           # 가중치 적용된 개별 점수
           weighted_scores = {
               'coherence': raw_scores['coherence'] * weights['coherence'],
               'distinctiveness': raw_scores['distinctiveness'] * weights['distinctiveness'],
               'diversity': raw_scores['diversity'] * weights['diversity']
           }

           # 최종 종합 점수
           overall_score = sum(weighted_scores.values())

           return {
               'coherence': coherence_result,
               'distinctiveness': distinctiveness_result,
               'diversity': diversity_result,
               'raw_scores': raw_scores,
               'weighted_scores': weighted_scores,
               'weights': weights,
               'overall_score': overall_score
           }

       except Exception as e:
           print(f"[ERROR] 토픽 모델 평가 중 오류 발생: {str(e)}")
           return {
               'coherence': self._get_default_evaluation_result(),
               'distinctiveness': self._get_default_evaluation_result(),
               'diversity': {'diversity': 0.0},
               'raw_scores': {'coherence': 0.0, 'distinctiveness': 0.0, 'diversity': 0.0},
               'weighted_scores': {'coherence': 0.0, 'distinctiveness': 0.0, 'diversity': 0.0},
               'weights': {'coherence': 0.5, 'distinctiveness': 0.3, 'diversity': 0.2},
               'overall_score': 0.0
           }

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
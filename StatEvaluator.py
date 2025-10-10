import numpy as np
import math
from typing import List, Dict, Any
from itertools import combinations
from gensim.models.coherencemodel import CoherenceModel
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

   def _evaluate_topic_diversity(self, topics: List[List[str]]) -> Dict[str, float]:
       # 토픽별 단어 출현 빈도 계산
       topic_word_freq = {}
       for topic_idx, topic in enumerate(topics):
           for word in topic:
               if word not in topic_word_freq:
                   topic_word_freq[word] = set()
               topic_word_freq[word].add(topic_idx)
       
       # 단어별 토픽 출현 수 기반 다양성 계산
       total_possible_occurrences = len(topics) * len(topic_word_freq)
       actual_occurrences = sum(len(topics_set) for topics_set in topic_word_freq.values())
       
       diversity = 1 - (actual_occurrences / total_possible_occurrences)
       return {'diversity_score': diversity}

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

   def evaluate(self, topics: List[List[str]], texts: List[List[str]], 
               dictionary, word_doc_freq: Dict[str, int],
               co_doc_freq: Dict[tuple, int], total_documents: int,
               vocab_size: int) -> Dict[str, Any]:
       """토픽 모델 평가 수행"""
       npmi_scores = self._evaluate_npmi(topics, word_doc_freq, co_doc_freq, total_documents)
       coherence_scores = self._evaluate_cv_coherence(topics, texts, dictionary)
       diversity_scores = self._evaluate_topic_diversity(topics)
       kld_scores = self._evaluate_kld(topics, vocab_size)
       jsd_scores = self._evaluate_jsd(topics, vocab_size)
       irbo_scores = self._evaluate_irbo(topics)
       
       return {
           'npmi': npmi_scores['average_npmi'],
           'coherence': coherence_scores['average_coherence'],
           'diversity': diversity_scores['diversity_score'],
           'kld': kld_scores['average_kld'],
           'jsd': jsd_scores['average_jsd'],
           'irbo': irbo_scores['irbo_score'],
           'overall_score': self._compute_overall_score(
               npmi_scores,
               coherence_scores,
               diversity_scores,
               kld_scores,
               jsd_scores,
               irbo_scores
           ),
           'detailed_scores': {
               'npmi': npmi_scores,
               'coherence': coherence_scores,
               'diversity': diversity_scores,
               'kld': kld_scores,
               'jsd': jsd_scores,
               'irbo': irbo_scores
           }
       }
# mini, small, medium data 사용

from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from .base_model import BaseTopicModel
import torch.nn.functional as F
import gc
import os

class CTEModel(BaseTopicModel, nn.Module):
    def __init__(self, num_topics: int):
        nn.Module.__init__(self)
        BaseTopicModel.__init__(self, num_topics=num_topics)
        
        self.num_topics = num_topics
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_cluster_size = 5
     
        self.word_doc_freq = defaultdict(int)
        self.co_doc_freq = defaultdict(int)
        self.vectorizer = None
        self.word_embeddings = {}
        self.embedding_dim = None
        self.topic_words = {}

    def set_word_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> None:
        """단어 임베딩 설정 및 평가기 초기화"""
        self.word_embeddings = embeddings
        if embeddings:
   
            self.embedding_dim = next(iter(embeddings.values())).shape[0]
        # BaseTopicModel의 평가기 초기화
        self.set_evaluators(embeddings)

    def get_word_embeddings(self) -> Dict[str, torch.Tensor]:
        return self.word_embeddings

    def get_embedding_dim(self) -> int:
        return 768

    def _cosine_similarity(self, word1: str, word2: str) -> float:
        """두 단어의 코사인 유사도 계산"""
        if word1 not in self.word_embeddings or word2 not in self.word_embeddings:
            return 0.0
            
        emb1 = self.word_embeddings[word1]
        emb2 = self.word_embeddings[word2]
        
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0)
        )
 
        return float(similarity.item())

    def fit(self, tokenized_texts: List[List[str]], embeddings: List[Any]) -> None:
        """모델 학습 구현"""
        try:
            print(f"[DEBUG] CTE 모델 학습 시작 (총 {len(embeddings)}개 문서)")
            
            # 1. 임베딩 준비 - 다양한 입력 형태 처리
            embeddings_array = []
            for emb in embeddings:
                if isinstance(emb, torch.Tensor):
                    embeddings_array.append(emb.cpu().numpy())
                elif isinstance(emb, np.ndarray):
                    embeddings_array.append(emb)
                elif isinstance(emb, list):
                    embeddings_array.append(np.array(emb))
                else:
                    raise ValueError(f"Unsupported embedding type: {type(emb)}")
            
            embeddings_array = np.array(embeddings_array)
            
            # 2. 클러스터링 수행
            print("[DEBUG] 클러스터링 수행 중...")
            kmeans = KMeans(n_clusters=self.num_topics, random_state=42)
            kmeans.fit(embeddings_array)
            self.cluster_labels = kmeans.labels_
            
            # 3. 벡터라이저 초기화
            print("[DEBUG] 벡터라이저 초기화 중...")
            self.vectorizer_model = TfidfVectorizer(
                tokenizer=lambda x: x,  
                lowercase=False,
                token_pattern=None  
            )
            
            # 4. TF-IDF 행렬 생성
            
            print("[DEBUG] TF-IDF 행렬 생성 중...")
            tfidf_matrix = self.vectorizer_model.fit_transform(tokenized_texts)
            feature_names = self.vectorizer_model.get_feature_names_out()
            
            # 5. 단어 출현 통계 수집
            print("[DEBUG] 단어 빈도 계산 중...")
            self.word_doc_freq.clear()
            self.co_doc_freq.clear()
            
            for tokens in tokenized_texts:
                # 문서 내 고유 단어에 대해서만 카운트
                unique_tokens = set(tokens)
                for word in unique_tokens:
                    self.word_doc_freq[word] += 1
                
                # 단어 쌍의 공출현 빈도 계산
                for i, word1 in enumerate(tokens):
                    for word2 in tokens[i+1:]:
                        if word1 < word2:
                            self.co_doc_freq[(word1, word2)] += 1
                        else:
                            self.co_doc_freq[(word2, word1)] += 1
            
            # 6. 토픽 키워드 추출
            self.topic_words = self.extract_topic_keywords(tokenized_texts, embeddings_array, feature_names, tfidf_matrix)
            
            print("[DEBUG] CTE 모델 학습 완료")
            self.fitted = True
           
        except Exception as e:
            print(f"[ERROR] CTE 모델 학습 중 치명적 오류: {str(e)}")
            raise

    def extract_topic_keywords(self, tokenized_texts, embeddings_array, feature_names, tfidf_matrix, batch_size=1000):
        """토픽별 키워드 추출 - 배치 처리로 메모리 효율성 확보"""
        print("[DEBUG] Extracting topic keywords with batched processing...")
        
        device = self.device
        num_words = len(feature_names)
        num_topics = self.num_topics
        
        # 임베딩을 텐서로 변환
        embeddings_tensor = torch.tensor(embeddings_array, device=device)
        print(f"[DEBUG] Embeddings shape: {embeddings_tensor.shape}")
        
        # 토픽별 키워드와 점수 저장
        topics = {}
        
        for topic_id in range(num_topics):
            print(f"[DEBUG] Processing topic {topic_id + 1}/{num_topics}")
            
            # 현재 토픽의 문서 인덱스
            topic_docs_indices = np.where(self.cluster_labels == topic_id)[0]
            if len(topic_docs_indices) < self.min_cluster_size:
                print(f"[WARNING] Topic {topic_id} has fewer documents than minimum size")
                continue
            
            # 현재 토픽의 문서 임베딩
            topic_docs = embeddings_tensor[topic_docs_indices]
            
            # 다른 토픽의 문서 임베딩
            other_docs_indices = np.where(self.cluster_labels != topic_id)[0]
            other_docs = embeddings_tensor[other_docs_indices] if len(other_docs_indices) > 0 else None
            
            # 토픽 중심점 계산
            topic_centroid = topic_docs.mean(dim=0)
            
            word_scores = {}
            
            # 배치 단위로 단어 처리
            for start_idx in range(0, len(self.word_embeddings), batch_size):
                try:
                    end_idx = min(start_idx + batch_size, len(self.word_embeddings))
                    current_words = list(self.word_embeddings.keys())[start_idx:end_idx]
                    
                    # 현재 배치의 단어 임베딩
                    batch_embeddings = torch.stack([
                        self.word_embeddings[word].to(device) 
                        for word in current_words 
                        if word in feature_names
                    ])
                    
                    if batch_embeddings.size(0) == 0:
                        continue
                    
                    # 1. 응집성: 토픽 내 문서들과의 유사도 (청크 단위로 계산)
                    topic_sim_scores = []
                    chunk_size = 50  # 메모리 사용 제한
                    for chunk_start in range(0, len(topic_docs), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(topic_docs))
                        topic_docs_chunk = topic_docs[chunk_start:chunk_end]

                        chunk_sim = F.cosine_similarity(
                            batch_embeddings.unsqueeze(1),
                            topic_docs_chunk.unsqueeze(0),
                            dim=2
                        ).mean(dim=1)
                        topic_sim_scores.append(chunk_sim)

                    topic_sim = torch.stack(topic_sim_scores).mean(dim=0)

                    # 2. 구별성: 다른 토픽과의 차별성 (청크 단위로 계산)
                    if other_docs is not None:
                        other_sim_scores = []
                        for chunk_start in range(0, len(other_docs), chunk_size):
                            chunk_end = min(chunk_start + chunk_size, len(other_docs))
                            other_docs_chunk = other_docs[chunk_start:chunk_end]

                            chunk_sim = F.cosine_similarity(
                                batch_embeddings.unsqueeze(1),
                                other_docs_chunk.unsqueeze(0),
                                dim=2
                            ).mean(dim=1)
                            other_sim_scores.append(chunk_sim)

                        other_sim = torch.stack(other_sim_scores).mean(dim=0)
                        distinctiveness = topic_sim - other_sim
                    else:
                        distinctiveness = topic_sim
                    
                    # 3. 대표성: 토픽 중심점과의 유사도
                    representativeness = F.cosine_similarity(
                        batch_embeddings,
                        topic_centroid.unsqueeze(0),
                        dim=1
                    )
                    
                    # 종합 점수 계산
                    batch_scores = (
                        0.4 * topic_sim +
                        0.3 * distinctiveness +
                        0.3 * representativeness
                    )
                    
                    # 결과 저장
                    for word, score in zip(current_words, batch_scores):
                        if word in feature_names:
                            word_scores[word] = score.item()
                    
                    # 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[WARNING] OOM in batch {start_idx}-{end_idx}, reducing batch size")
                        smaller_batch_size = batch_size // 2
                        if smaller_batch_size < 100:
                            raise RuntimeError("Batch size too small, cannot proceed")
                        return self.extract_topic_keywords(
                            tokenized_texts, embeddings_array, feature_names,
                            tfidf_matrix, batch_size=smaller_batch_size
                        )
                    else:
                        raise e
            
            # 상위 키워드 선택
            final_word_scores = []
            seen_words = set()
            
            for word, score in sorted(word_scores.items(), key=lambda x: x[1], reverse=True):
                if word not in seen_words:
                    final_word_scores.append((word, score))
                    seen_words.add(word)
                if len(final_word_scores) >= 50:  # 상위 50개 키워드 선택
                    break
            
            topics[topic_id] = final_word_scores
        
        return topics

    def get_topics(self, top_n: int = 20) -> List[List[str]]:
        """토픽 및 키워드 결과 반환

        Args:
            top_n: 각 토픽당 반환할 상위 키워드 수 (기본값: 20)

        Returns:
            토픽별 키워드 리스트
        """
        try:
            topics_list = []
            for idx in range(self.num_topics):
                if idx in self.topic_words and len(self.topic_words[idx]) > 0:
                    try:
                        words = [word for word, _ in self.topic_words[idx][:top_n]]
                        topics_list.append(words)
                    except:
                        topics_list.append(['unknown'])
                else:
                    topics_list.append(['unknown'])
            
            return topics_list
        except Exception as e:
            print(f"Error in get_topics: {e}")
            return [['unknown'] for _ in range(self.num_topics)]

    def get_document_topics(self, embeddings: List[Any]) -> List[int]:
        """문서별 토픽 할당 반환

        Args:
            embeddings: 문서 임베딩 리스트 (fit에서 사용한 것과 동일)

        Returns:
            문서별 토픽 인덱스 리스트
        """
        if not hasattr(self, 'cluster_labels'):
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        return self.cluster_labels.tolist()
            
    def get_model_stats(self) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다")
        
        try:
            # 토픽별 문서 수 계산
            topic_sizes = {}
            if hasattr(self, 'cluster_labels') and self.cluster_labels is not None:
                for topic_id in range(self.num_topics):
                    topic_docs = np.where(self.cluster_labels == topic_id)[0]
                    topic_sizes[topic_id] = len(topic_docs)
            else:
                topic_sizes = {i: 0 for i in range(self.num_topics)}
                
            # 토픽별 키워드 수
            topic_keyword_counts = {}
            if self.topic_words:
                for topic_id, keywords in self.topic_words.items():
                    topic_keyword_counts[topic_id] = len(keywords)
                    
            return {
                'num_topics': self.num_topics,
                'topic_sizes': topic_sizes,
                'topic_keyword_counts': topic_keyword_counts,
                'total_documents': len(self.cluster_labels) if hasattr(self, 'cluster_labels') else 0,
                'vocabulary_size': len(self.vectorizer_model.get_feature_names_out()) if self.vectorizer_model else 0,
                'word_doc_freq': dict(self.word_doc_freq),
                'co_doc_freq': dict(self.co_doc_freq)
            }
        except Exception as e:
            print(f"[ERROR] 모델 통계 생성 중 오류: {str(e)}")
            return {
                'num_topics': self.num_topics,
                'topic_sizes': {i: 0 for i in range(self.num_topics)},
                'topic_keyword_counts': {},
                'total_documents': 0,
                'vocabulary_size': 0,
                'word_doc_freq': {},
                'co_doc_freq': {}
            }
            
    def _cleanup_memory(self):
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 임시 파일 삭제
        if hasattr(self, 'temp_files'):
            for file_path in self.temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)




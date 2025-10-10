# 수정된 임포트
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1. 데이터 불러오기
df_distinct = pd.read_csv('data/distinct_topic.csv')
df_similar = pd.read_csv('data/similar_topic.csv')
df_more_similar = pd.read_csv('data/more_similar_topic.csv')

# 2. TF-IDF 벡터라이저 정의
def create_tfidf_vectorizer():
    """TF-IDF 벡터라이저 생성"""
    return TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

# 3. 토픽별 키워드 추출 (TF-IDF 기반)
def get_top_keywords_tfidf(texts, n=10):
    """TF-IDF 기반 키워드 추출"""
    tfidf = create_tfidf_vectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()
    avg_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = avg_tfidf.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_indices]

# 4. 문서-토픽 TF-IDF 행렬 생성
def get_tfidf_matrix(texts):
    """문서들의 TF-IDF 행렬 생성"""
    tfidf = create_tfidf_vectorizer()
    return tfidf.fit_transform(texts)

# Clear 데이터의 TF-IDF 행렬 생성
tfidf_matrix_distinct = get_tfidf_matrix(df_distinct['text'])
tfidf_matrix_similar = get_tfidf_matrix(df_similar['text'])
tfidf_matrix_more_similar = get_tfidf_matrix(df_more_similar['text'])

# 5. t-SNE를 사용한 2D 시각화 (Distinct vs. Similar, Similar vs. More_Similar)
tsne_results = {}
for name, matrix in [('distinct', tfidf_matrix_distinct), 
                    ('similar', tfidf_matrix_similar),
                    ('more_similar', tfidf_matrix_more_similar)]:
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results[name] = tsne.fit_transform(matrix.toarray())

# 6. 토픽별 키워드 추출
topics_more_similar = df_more_similar['label'].unique()
keywords_more_similar = {topic: get_top_keywords_tfidf(df_more_similar[df_more_similar['label'] == topic]['text']) 
                    for topic in topics_more_similar}

# Define topics_similar before visualization
topics_similar = df_similar['label'].unique()

# Define keywords for distinct topics (add this before visualization)
topics_distinct = df_distinct['label'].unique()
keywords = {topic: get_top_keywords_tfidf(df_distinct[df_distinct['label'] == topic]['text']) 
           for topic in topics_distinct}

# Define keywords for similar topics (add this before visualization)
keywords_similar = {topic: get_top_keywords_tfidf(df_similar[df_similar['label'] == topic]['text']) 
                for topic in topics_similar}

# 7. 시각화
# 7. 시각화 수정
# First comparison: Distinct vs Similar
fig1, axes1 = plt.subplots(1, 2, figsize=(24, 8))

# Left plot - Distinct
colors_distinct = plt.cm.rainbow(np.linspace(0, 1, len(topics_distinct)))
for i, topic in enumerate(topics_distinct):
    mask = df_distinct['label'] == topic
    axes1[0].scatter(tsne_results['distinct'][mask, 0], tsne_results['distinct'][mask, 1], 
                    c=[colors_distinct[i]], label=f"Distinct-{topic}", alpha=0.7)
axes1[0].set_title('Distinct Topics Distribution')
axes1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Right plot - Similar
colors_similar = plt.cm.rainbow(np.linspace(0, 1, len(topics_similar)))
for i, topic in enumerate(topics_similar):
    mask = df_similar['label'] == topic
    axes1[1].scatter(tsne_results['similar'][mask, 0], tsne_results['similar'][mask, 1], 
                    c=[colors_similar[i]], label=f"Similar-{topic}", alpha=0.7)
axes1[1].set_title('Similar Topics Distribution')
axes1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Comparison: Distinct vs Similar Topic Distributions', fontsize=16)
plt.tight_layout()
plt.show()

# Second comparison: Similar vs More Similar
fig2, axes2 = plt.subplots(1, 2, figsize=(24, 8))

# Left plot - Similar
for i, topic in enumerate(topics_similar):
    mask = df_similar['label'] == topic
    axes2[0].scatter(tsne_results['similar'][mask, 0], tsne_results['similar'][mask, 1], 
                    c=[colors_similar[i]], label=f"Similar-{topic}", alpha=0.7)
axes2[0].set_title('Similar Topics Distribution')
axes2[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Right plot - More Similar
colors_more_similar = plt.cm.rainbow(np.linspace(0, 1, len(topics_more_similar)))
for i, topic in enumerate(topics_more_similar):
    mask = df_more_similar['label'] == topic
    axes2[1].scatter(tsne_results['more_similar'][mask, 0], tsne_results['more_similar'][mask, 1], 
                    c=[colors_more_similar[i]], label=f"More_Similar-{topic}", alpha=0.7)
axes2[1].set_title('More Similar Topics Distribution')
axes2[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Comparison: Similar vs More Similar Topic Distributions', fontsize=16)
plt.tight_layout()
plt.show()

# 7.2 키워드 시각화 - 표 형식으로 변경
print("\nDistinct Topics Keywords:")
distinct_keywords_df = pd.DataFrame([(topic, ', '.join(words)) for topic, words in keywords.items()],
                                  columns=['Topic', 'Keywords'])
print(distinct_keywords_df.to_string(index=False))

print("\nSimilar Topics Keywords:")
similar_keywords_df = pd.DataFrame([(topic, ', '.join(words)) for topic, words in keywords_similar.items()],
                                 columns=['Topic', 'Keywords'])
print(similar_keywords_df.to_string(index=False))

print("\nMore Similar Topics Keywords:")
more_similar_keywords_df = pd.DataFrame([(topic, ', '.join(words)) for topic, words in keywords_more_similar.items()],
                                      columns=['Topic', 'Keywords'])
print(more_similar_keywords_df.to_string(index=False))

# 7.3 토픽 유사도 시각화
def get_topic_similarity_matrix(df, tfidf_matrix):
    """토픽 간 유사도 행렬 계산"""
    topics = df['label'].unique()
    similarity_matrix = np.zeros((len(topics), len(topics)))
    
    for i, topic1 in enumerate(topics):
        for j, topic2 in enumerate(topics):
            mask1 = df['label'] == topic1
            mask2 = df['label'] == topic2
            
            # sparse matrix를 dense array로 변환
            topic1_vectors = tfidf_matrix[mask1].mean(axis=0).A
            topic2_vectors = tfidf_matrix[mask2].mean(axis=0).A
            
            similarity = cosine_similarity(topic1_vectors, topic2_vectors)[0][0]
            similarity_matrix[i, j] = similarity
            
    return similarity_matrix

# 8. 결과 저장
# 8.1 TF-IDF 행렬 저장
with open('data/tfidf_matrix_distinct.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix_distinct, f)

with open('data/tfidf_matrix_similar.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix_similar, f)

# 8.2 토픽 키워드 저장
topics_distinct = [get_top_keywords_tfidf(df_distinct[df_distinct['label'] == topic]['text']) 
                for topic in df_distinct['label'].unique()]

topics_similar = [get_top_keywords_tfidf(df_similar[df_similar['label'] == topic]['text']) 
               for topic in df_similar['label'].unique()]

with open('data/topics_distinct_tfidf.pkl', 'wb') as f:
    pickle.dump(topics_distinct, f)

with open('data/topics_similar_tfidf.pkl', 'wb') as f:
    pickle.dump(topics_similar, f)

# 8.3 문서 텍스트 저장
with open('data/docs_distinct_tfidf.pkl', 'wb') as f:
    pickle.dump(df_distinct['text'].tolist(), f)

with open('data/docs_similar_tfidf.pkl', 'wb') as f:
    pickle.dump(df_similar['text'].tolist(), f)

# More_Similar 데이터 추가 저장
with open('data/tfidf_matrix_more_similar.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix_more_similar, f)

topics_more_similar = [get_top_keywords_tfidf(df_more_similar[df_more_similar['label'] == topic]['text']) 
                   for topic in df_more_similar['label'].unique()]

with open('data/topics_more_similar_tfidf.pkl', 'wb') as f:
    pickle.dump(topics_more_similar, f)

with open('data/docs_more_similar_tfidf.pkl', 'wb') as f:
    pickle.dump(df_more_similar['text'].tolist(), f)

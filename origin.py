import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. 데이터 불러오기
df_distinct = pd.read_csv('data/distinct_topic.csv')
df_similar = pd.read_csv('data/similar_topic.csv')
df_more_similar = pd.read_csv('data/more_similar_topic.csv')

# 2. 문서 임베딩
def get_embeddings(df, filename):
    try:
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
    except FileNotFoundError:
        embeddings = model.encode(df['text'].tolist())
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

# Load or compute embeddings
embeddings_distinct = get_embeddings(df_distinct, 'data/embeddings_distinct.pkl')
embeddings_similar = get_embeddings(df_similar, 'data/embeddings_similar.pkl')
embeddings_more_similar = get_embeddings(df_more_similar, 'data/embeddings_more_similar.pkl')

# 3. 토픽별 키워드 추출 (임베딩 기반)
def get_top_keywords_embedding(texts, embeddings, n=10):
    vectorizer = CountVectorizer().fit(texts)
    words = vectorizer.get_feature_names_out()
    word_embeddings = model.encode(words)
    
    avg_doc_embedding = np.mean(embeddings, axis=0)
    similarities = cosine_similarity([avg_doc_embedding], word_embeddings)[0]
    
    top_idx = similarities.argsort()[-n:][::-1]
    return [words[i] for i in top_idx]

# Clear 데이터의 토픽별 키워드 추출
topics_distinct = df_distinct['label'].unique()
keywords_distinct = {topic: get_top_keywords_embedding(df_distinct[df_distinct['label'] == topic]['text'], 
                                                     embeddings_distinct[df_distinct['label'] == topic]) 
                    for topic in topics_distinct}

# Similar 데이터의 토픽별 키워드 추출
topics_similar = df_similar['label'].unique()
keywords_similar = {topic: get_top_keywords_embedding(df_similar[df_similar['label'] == topic]['text'], 
                                                    embeddings_similar[df_similar['label'] == topic]) 
                   for topic in topics_similar}

# More Similar 데이터의 토픽별 키워드 추출
topics_more_similar = df_more_similar['label'].unique()
keywords_more_similar = {topic: get_top_keywords_embedding(df_more_similar[df_more_similar['label'] == topic]['text'], 
                                                         embeddings_more_similar[df_more_similar['label'] == topic]) 
                        for topic in topics_more_similar}



# 4. t-SNE를 사용한 2D 시각화
tsne_distinct = TSNE(n_components=2, random_state=42)
tsne_results_distinct = tsne_distinct.fit_transform(embeddings_distinct)

tsne_similar = TSNE(n_components=2, random_state=42)
tsne_results_similar = tsne_similar.fit_transform(embeddings_similar)

# 5. Similar vs More Similar t-SNE 시각화
tsne_similar_compare = TSNE(n_components=2, random_state=42)
tsne_results_similar_compare = tsne_similar_compare.fit_transform(embeddings_similar)

tsne_more_similar = TSNE(n_components=2, random_state=42)
tsne_results_more_similar = tsne_more_similar.fit_transform(embeddings_more_similar)

# 4. t-SNE Visualizations
# First visualization: Distinct vs Similar
fig1, axes1 = plt.subplots(1, 2, figsize=(24, 8))

# Distinct data visualization
topics_distinct = df_distinct['label'].unique()
colors_distinct = plt.cm.rainbow(np.linspace(0, 1, len(topics_distinct)))
for i, topic in enumerate(topics_distinct):
    mask = df_distinct['label'] == topic
    axes1[0].scatter(tsne_results_distinct[mask, 0], tsne_results_distinct[mask, 1], 
                    c=[colors_distinct[i]], label=topic, alpha=0.7)
axes1[0].set_title('t-SNE Visualization of Distinct Topics')
axes1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Similar data visualization
topics_similar = df_similar['label'].unique()
colors_similar = plt.cm.rainbow(np.linspace(0, 1, len(topics_similar)))
for i, topic in enumerate(topics_similar):
    mask = df_similar['label'] == topic
    axes1[1].scatter(tsne_results_similar[mask, 0], tsne_results_similar[mask, 1], 
                    c=[colors_similar[i]], label=topic, alpha=0.7)
axes1[1].set_title('t-SNE Visualization of Similar Topics')
axes1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Second visualization: Similar vs More Similar
fig2, axes2 = plt.subplots(1, 2, figsize=(24, 8))

# Similar data visualization
for i, topic in enumerate(topics_similar):
    mask = df_similar['label'] == topic
    axes2[0].scatter(tsne_results_similar_compare[mask, 0], tsne_results_similar_compare[mask, 1], 
                    c=[colors_similar[i]], label=topic, alpha=0.7)
axes2[0].set_title('t-SNE Visualization of Similar Topics')
axes2[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# More Similar data visualization
topics_more_similar = df_more_similar['label'].unique()
colors_more_similar = plt.cm.rainbow(np.linspace(0, 1, len(topics_more_similar)))
for i, topic in enumerate(topics_more_similar):
    mask = df_more_similar['label'] == topic
    axes2[1].scatter(tsne_results_more_similar[mask, 0], tsne_results_more_similar[mask, 1], 
                    c=[colors_more_similar[i]], label=topic, alpha=0.7)
axes2[1].set_title('t-SNE Visualization of More Similar Topics')
axes2[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# 6. 토픽별 키워드를 표 형식으로 출력
def display_keywords_table(keywords_dict, title):
    df_keywords = pd.DataFrame(
        [(topic, ', '.join(words)) for topic, words in keywords_dict.items()],
        columns=['Topic', 'Keywords']
    )
    
    print(f"\n{title}")
    print("=" * 100)
    print(df_keywords.to_string(index=False))
    print("=" * 100)

# Display keywords for all datasets
print("\nKeyword Analysis Results")
print("=" * 100)

display_keywords_table(keywords_distinct, "Distinct Topics Keywords")
display_keywords_table(keywords_similar, "Similar Topics Keywords")
display_keywords_table(keywords_more_similar, "More Similar Topics Keywords")

# 8. 각 데이터에 대한 기본 통계
def display_dataset_statistics(df, embeddings, dataset_name):
    print(f"\n{dataset_name} Dataset Statistics")
    print("=" * 100)
    
    # 1. 전체 문서 수
    total_docs = len(df)
    print(f"Total Documents: {total_docs}")
    
    # 2. 토픽별 문서 수와 비율
    topic_counts = df['label'].value_counts()
    topic_percentages = (topic_counts / total_docs * 100).round(2)
    print("\nDocuments per Topic:")
    for topic in topic_counts.index:
        print(f"{topic}: {topic_counts[topic]} documents ({topic_percentages[topic]}%)")
    
    # 3. 문서 길이 통계
    doc_lengths = df['text'].str.split().str.len()
    print(f"\nDocument Length Statistics:")
    print(f"Average Length: {doc_lengths.mean():.2f} words")
    print(f"Median Length: {doc_lengths.median():.2f} words")
    print(f"Min Length: {doc_lengths.min()} words")
    print(f"Max Length: {doc_lengths.max()} words")
    
    # 4. 토픽 간 평균 코사인 유사도
    topics = df['label'].unique()
    topic_embeddings = {topic: np.mean(embeddings[df['label'] == topic], axis=0) for topic in topics}
    similarities = cosine_similarity(list(topic_embeddings.values()))
    avg_similarity = (similarities.sum() - len(topics)) / (len(topics) * (len(topics) - 1))
    print(f"\nAverage Inter-topic Similarity: {avg_similarity:.4f}")
    
    print("=" * 100)

# Distinct 데이터 통계
display_dataset_statistics(df_distinct, embeddings_distinct, "Distinct")

# Similar 데이터 통계
display_dataset_statistics(df_similar, embeddings_similar, "Similar")

# More Similar 데이터 통계
display_dataset_statistics(df_more_similar, embeddings_more_similar, "More Similar")


# Extract keywords and create embedding cache
def create_embedding_cache(df, embeddings):
    embedding_cache = {}
    topics = df['label'].unique()
    for topic in topics:
        topic_texts = df[df['label'] == topic]['text']
        keywords = get_top_keywords_embedding(topic_texts, embeddings[df['label'] == topic])
        cache_key = ' '.join(keywords)
        avg_embedding = np.mean(embeddings[df['label'] == topic], axis=0)
        embedding_cache[cache_key] = avg_embedding
    return embedding_cache

# Create embedding cache for all datasets
embedding_cache_distinct = create_embedding_cache(df_distinct, embeddings_distinct)
embedding_cache_similar = create_embedding_cache(df_similar, embeddings_similar)
embedding_cache_more_similar = create_embedding_cache(df_more_similar, embeddings_more_similar)

# Save the embedding cache
with open('data/embedding_cache_distinct.pkl', 'wb') as f:
    pickle.dump(embedding_cache_distinct, f)

with open('data/embedding_cache_similar.pkl', 'wb') as f:
    pickle.dump(embedding_cache_similar, f)

with open('data/embedding_cache_more_similar.pkl', 'wb') as f:
    pickle.dump(embedding_cache_more_similar, f)

# Save topics and documents
topics_distinct = [get_top_keywords_embedding(df_distinct[df_distinct['label'] == topic]['text'], 
                                           embeddings_distinct[df_distinct['label'] == topic]) 
                  for topic in df_distinct['label'].unique()]

topics_similar = [get_top_keywords_embedding(df_similar[df_similar['label'] == topic]['text'], 
                                          embeddings_similar[df_similar['label'] == topic]) 
                 for topic in df_similar['label'].unique()]

topics_more_similar = []  # 한 번만 정의
for topic in df_more_similar['label'].unique():
    topic_mask = df_more_similar['label'] == topic
    topic_texts = df_more_similar[topic_mask]['text']
    topic_embeddings = embeddings_more_similar[topic_mask]
    keywords = get_top_keywords_embedding(topic_texts, topic_embeddings)
    topics_more_similar.append(keywords)

# Save all topics
with open('data/topics_distinct.pkl', 'wb') as f:
    pickle.dump(topics_distinct, f)

with open('data/topics_similar.pkl', 'wb') as f:
    pickle.dump(topics_similar, f)

with open('data/topics_more_similar.pkl', 'wb') as f:
    pickle.dump(topics_more_similar, f)

# Save documents
docs_distinct = df_distinct['text'].tolist()
docs_similar = df_similar['text'].tolist()
docs_more_similar = df_more_similar['text'].tolist()

with open('data/docs_distinct.pkl', 'wb') as f:
    pickle.dump(docs_distinct, f)

with open('data/docs_similar.pkl', 'wb') as f:
    pickle.dump(docs_similar, f)

with open('data/docs_more_similar.pkl', 'wb') as f:
    pickle.dump(docs_more_similar, f)

# Initialize BERT tokenizer and model
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

def process_with_bert(texts, batch_size=32):
    """Process texts with BERT and save tokenized outputs"""
    all_outputs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = bert_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            batch_outputs = {
                'input_ids': inputs['input_ids'].cpu().numpy(),
                'attention_mask': inputs['attention_mask'].cpu().numpy(),
                'last_hidden_state': outputs.last_hidden_state.cpu().numpy()
            }
            all_outputs.append(batch_outputs)
    
    return all_outputs

# Process all datasets
bert_outputs_distinct = process_with_bert(docs_distinct)
bert_outputs_similar = process_with_bert(docs_similar)
bert_outputs_more_similar = process_with_bert(docs_more_similar)

# Save processed BERT outputs
with open('data/bert_outputs_distinct.pkl', 'wb') as f:
    pickle.dump(bert_outputs_distinct, f)

with open('data/bert_outputs_similar.pkl', 'wb') as f:
    pickle.dump(bert_outputs_similar, f)

with open('data/bert_outputs_more_similar.pkl', 'wb') as f:
    pickle.dump(bert_outputs_more_similar, f)


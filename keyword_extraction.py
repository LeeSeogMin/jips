"""
Advanced Keyword Extraction and Topic Analysis Tool

This script provides advanced keyword extraction and topic analysis functionality:
1. Embedding-based keyword extraction
2. Topic visualization using t-SNE
3. Inter-topic similarity analysis
4. BERT-based document processing
"""

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
import os
from typing import Dict, List, Tuple, Any

class TopicAnalyzer:
    def __init__(self, cache_dir: str = 'data'):
        """
        Initialize the Topic Analyzer
        
        Args:
            cache_dir (str): Directory for caching embeddings and results
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize models
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERT models
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
    def load_cached_embeddings(self, cache_path: str, texts: List[str]) -> np.ndarray:
        """
        Load embeddings from cache if available, otherwise generate and cache them
        """
        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
        except FileNotFoundError:
            embeddings = self.sentence_transformer.encode(texts, show_progress_bar=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
        return embeddings
    
    def extract_keywords_embedding(self, texts: List[str], embeddings: np.ndarray, n: int = 10) -> List[str]:
        """
        Extract keywords using embedding-based similarity
        """
        vectorizer = CountVectorizer(
            min_df=2,
            max_df=0.95,
            stop_words='english'
        ).fit(texts)
        words = vectorizer.get_feature_names_out()
        
        word_embeddings = self.sentence_transformer.encode(words, show_progress_bar=True)
        avg_doc_embedding = np.mean(embeddings, axis=0)
        similarities = cosine_similarity([avg_doc_embedding], word_embeddings)[0]
        
        top_idx = similarities.argsort()[-n:][::-1]
        return [words[i] for i in top_idx]
    
    def extract_topic_keywords(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, List[str]]:
        """
        Extract keywords for each topic in the dataset
        """
        topics = df['label'].unique()
        keywords = {}
        
        for topic in topics:
            mask = df['label'] == topic
            topic_texts = df[mask]['text'].tolist()
            topic_embeddings = embeddings[mask]
            keywords[topic] = self.extract_keywords_embedding(topic_texts, topic_embeddings)
        
        return keywords
    
    def visualize_topics_tsne(self, embeddings: np.ndarray, labels: np.ndarray, title: str) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create t-SNE visualization of topic embeddings
        """
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                      c=[colors[i]], label=label, alpha=0.7)
        
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig, tsne_results
    
    def analyze_dataset_statistics(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Analyze dataset statistics and inter-topic similarities
        """
        stats = {
            'total_documents': len(df),
            'topic_counts': df['label'].value_counts().to_dict(),
            'doc_length_stats': {
                'mean': df['text'].str.split().str.len().mean(),
                'median': df['text'].str.split().str.len().median(),
                'min': df['text'].str.split().str.len().min(),
                'max': df['text'].str.split().str.len().max()
            }
        }
        
        # Calculate inter-topic similarities
        topics = df['label'].unique()
        topic_embeddings = {topic: np.mean(embeddings[df['label'] == topic], axis=0) for topic in topics}
        similarities = cosine_similarity(list(topic_embeddings.values()))
        avg_similarity = (similarities.sum() - len(topics)) / (len(topics) * (len(topics) - 1))
        stats['inter_topic_similarity'] = avg_similarity
        
        return stats
    
    def process_with_bert(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, np.ndarray]]:
        """
        Process texts with BERT and return tokenized outputs
        """
        all_outputs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                batch_outputs = {
                    'input_ids': inputs['input_ids'].cpu().numpy(),
                    'attention_mask': inputs['attention_mask'].cpu().numpy(),
                    'last_hidden_state': outputs.last_hidden_state.cpu().numpy()
                }
                all_outputs.append(batch_outputs)
        
        return all_outputs
    
    def create_embedding_cache(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create and return embedding cache for topics
        """
        cache = {}
        topics = df['label'].unique()
        
        for topic in topics:
            mask = df['label'] == topic
            topic_texts = df[mask]['text'].tolist()
            keywords = self.extract_keywords_embedding(topic_texts, embeddings[mask])
            cache_key = ' '.join(keywords)
            cache[cache_key] = np.mean(embeddings[mask], axis=0)
        
        return cache

def main():
    """
    Main function to demonstrate usage
    """
    analyzer = TopicAnalyzer()
    
    # Load datasets
    datasets = {
        'distinct': pd.read_csv('data/distinct_topic.csv'),
        'similar': pd.read_csv('data/similar_topic.csv'),
        'more_similar': pd.read_csv('data/more_similar_topic.csv')
    }
    
    for name, df in datasets.items():
        print(f"\nProcessing {name} dataset...")
        
        # Generate embeddings
        embeddings = analyzer.load_cached_embeddings(
            f'data/embeddings_{name}.pkl',
            df['text'].tolist()
        )
        
        # Extract keywords
        keywords = analyzer.extract_topic_keywords(df, embeddings)
        
        # Create visualization
        fig, _ = analyzer.visualize_topics_tsne(
            embeddings,
            df['label'].values,
            f't-SNE Visualization of {name.title()} Topics'
        )
        fig.savefig(f'data/tsne_{name}.png')
        plt.close(fig)
        
        # Analyze statistics
        stats = analyzer.analyze_dataset_statistics(df, embeddings)
        
        # Process with BERT
        bert_outputs = analyzer.process_with_bert(df['text'].tolist())
        
        # Save results
        with open(f'data/keywords_{name}.pkl', 'wb') as f:
            pickle.dump(keywords, f)
        
        with open(f'data/stats_{name}.pkl', 'wb') as f:
            pickle.dump(stats, f)
        
        with open(f'data/bert_{name}.pkl', 'wb') as f:
            pickle.dump(bert_outputs, f)
        
        print(f"Completed processing {name} dataset")

if __name__ == '__main__':
    main()
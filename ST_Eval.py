import numpy as np
import json
import pandas as pd
import pickle
from StatEvaluator import TopicModelStatEvaluator  
from gensim.corpora import Dictionary
from multiprocessing import freeze_support

def main():
    # 1. Load data
    df_distinct = pd.read_csv('data/distinct_topic.csv')
    df_similar = pd.read_csv('data/similar_topic.csv')
    df_more_similar = pd.read_csv('data/more_similar_topic.csv')

    # 2. Load TF-IDF derived keywords for each topic
    with open('data/topics_distinct_tfidf.pkl', 'rb') as f:
        topic_keywords_distinct = pickle.load(f)
        
    with open('data/topics_similar_tfidf.pkl', 'rb') as f:
        topic_keywords_similar = pickle.load(f)

    with open('data/topics_more_similar_tfidf.pkl', 'rb') as f:
        topic_keywords_more_similar = pickle.load(f)

    print("Loaded Keywords Preview:")
    print("Distinct Topics Count:", len(topic_keywords_distinct))
    print("Sample Distinct Topics Keywords:", topic_keywords_distinct[:2])
    print("\nSimilar Topics Count:", len(topic_keywords_similar))
    print("Sample Similar Topics Keywords:", topic_keywords_similar[:2])
    print("\nMore Similar Topics Count:", len(topic_keywords_more_similar))
    print("Sample More Similar Topics Keywords:", topic_keywords_more_similar[:2])

    # Add topic keywords visualization
    print("\nTopic Keywords for Each Dataset")
    print("=" * 70)

    def create_topic_keyword_df(topic_keywords, title):
        # Convert topic keywords list to DataFrame
        df = pd.DataFrame(topic_keywords)
        # Add topic numbers as index
        df.index = [f'Topic {i+1}' for i in range(len(topic_keywords))]
        # Add column names
        df.columns = [f'Keyword {i+1}' for i in range(df.shape[1])]
        
        print(f"\n{title}")
        print("-" * 70)
        print(df.to_string(justify='left'))

    create_topic_keyword_df(topic_keywords_distinct, "Distinct Topics Keywords")
    create_topic_keyword_df(topic_keywords_similar, "Similar Topics Keywords")
    create_topic_keyword_df(topic_keywords_more_similar, "More Similar Topics Keywords")

    # 3. Prepare text data for evaluation
    texts_distinct = [text.split() for text in df_distinct['text']]
    texts_similar = [text.split() for text in df_similar['text']]
    texts_more_similar = [text.split() for text in df_more_similar['text']]

    dictionary_distinct = Dictionary(texts_distinct)
    dictionary_similar = Dictionary(texts_similar)
    dictionary_more_similar = Dictionary(texts_more_similar)

    def get_word_frequencies(texts):
        word_doc_freq = {}
        co_doc_freq = {}
        
        for doc in texts:
            doc_words = set(doc)
            for word in doc_words:
                word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
            
            for word1 in doc_words:
                for word2 in doc_words:
                    if word1 < word2:
                        pair = (word1, word2)
                        co_doc_freq[pair] = co_doc_freq.get(pair, 0) + 1
        
        return word_doc_freq, co_doc_freq

    word_freq_distinct, coword_freq_distinct = get_word_frequencies(texts_distinct)
    word_freq_similar, coword_freq_similar = get_word_frequencies(texts_similar)
    word_freq_more_similar, coword_freq_more_similar = get_word_frequencies(texts_more_similar)

    # Initialize the evaluator
    evaluator = TopicModelStatEvaluator()

    # 4. Perform evaluation using TF-IDF derived keywords (5 times)
    num_iterations = 5
    eval_results = {
        'distinct': [],
        'similar': [],
        'more_similar': []
    }

    print("\nPerforming evaluations...")
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        # Evaluate distinct topics
        eval_distinct = evaluator.evaluate(
            topics=topic_keywords_distinct,
            texts=texts_distinct,
            dictionary=dictionary_distinct,
            word_doc_freq=word_freq_distinct,
            co_doc_freq=coword_freq_distinct,
            total_documents=len(texts_distinct),
            vocab_size=len(dictionary_distinct)
        )
        eval_results['distinct'].append(eval_distinct)

        # Evaluate similar topics
        eval_similar = evaluator.evaluate(
            topics=topic_keywords_similar,
            texts=texts_similar,
            dictionary=dictionary_similar,
            word_doc_freq=word_freq_similar,
            co_doc_freq=coword_freq_similar,
            total_documents=len(texts_similar),
            vocab_size=len(dictionary_similar)
        )
        eval_results['similar'].append(eval_similar)

        # Evaluate more similar topics
        eval_more_similar = evaluator.evaluate(
            topics=topic_keywords_more_similar,
            texts=texts_more_similar,
            dictionary=dictionary_more_similar,
            word_doc_freq=word_freq_more_similar,
            co_doc_freq=coword_freq_more_similar,
            total_documents=len(texts_more_similar),
            vocab_size=len(dictionary_more_similar)
        )
        eval_results['more_similar'].append(eval_more_similar)

    # Calculate average results for each dataset
    def calculate_avg_results(eval_list):
        """
        메트릭별 평균 결과 계산 및 정리
        Args:
            eval_list: 평가 결과 리스트
        Returns:
            평균 메트릭 결과 딕셔너리
        """
        # 평가할 메트릭 정의
        metrics = {
            'npmi': 'Distinctiveness',
            'coherence': 'Semantic Coherence',
            'diversity': 'Topic Diversity',
            'kld': 'KL Divergence',
            'jsd': 'Jensen-Shannon Distance',
            'irbo': 'Semantic Integration',
            'overall_score': 'Overall Score'
        }
        
        # 각 메트릭별 평균 계산
        avg_results = {}
        for metric in metrics.keys():
            values = [e[metric] for e in eval_list]
            avg_results[metric] = np.mean(values)
        
        return avg_results

    avg_distinct = calculate_avg_results(eval_results['distinct'])
    avg_similar = calculate_avg_results(eval_results['similar'])
    avg_more_similar = calculate_avg_results(eval_results['more_similar'])

    # 5. Print individual evaluation results (using averages)
    print("\nAverage Evaluation Results (5 iterations)")
    print("\nEvaluation Results for Distinct Topics:")
    for metric, value in avg_distinct.items():
        if metric != 'detailed_scores':
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

    print("\nEvaluation Results for Similar Topics:")
    for metric, value in avg_similar.items():
        if metric != 'detailed_scores':
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

    print("\nEvaluation Results for More Similar Topics:")
    for metric, value in avg_more_similar.items():
        if metric != 'detailed_scores':
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

    # Create comparison table with averaged results
    metrics = {
        'npmi': 'Distinctiveness',
        'coherence': 'Semantic Coherence',
        'diversity': 'Topic Diversity',
        'kld': 'KL Divergence',
        'jsd': 'Jensen-Shannon Distance',
        'irbo': 'Semantic Integration',
        'overall_score': 'Overall Score'
    }

    comparison_data = {
        'Distinct': [avg_distinct[m] for m in metrics.keys()],
        'Similar': [avg_similar[m] for m in metrics.keys()],
        'More Similar': [avg_more_similar[m] for m in metrics.keys()]
    }

    comparison_df = pd.DataFrame(
        comparison_data,
        index=[v for v in metrics.values()]
    )

    print("\nDataset Comparison (Average Metrics)")
    print("=" * 60)
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    print(comparison_df.to_string())

    # 6. Create comparison analysis
    print("\nDetailed Comparison:")
    for metric in metrics.keys():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"Distinct Topics: {avg_distinct[metric]:.3f}")
        print(f"Similar Topics: {avg_similar[metric]:.3f}")
        print(f"More Similar Topics: {avg_more_similar[metric]:.3f}")

    # 7. Create comparison table
    print("\nComparison Table:")
    # 포맷팅을 위한 float_format 추가
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(comparison_df.to_string())

    # 8. Save results
    comparison_df.to_csv('stat_evaluation_comparison.csv')

    # detailed_scores가 없으므로 기본 메트릭 결과만 저장
    with open('stat_evaluation_details.json', 'w') as f:
        json.dump({
            'distinct_topics': avg_distinct,
            'similar_topics': avg_similar,
            'more_similar_topics': avg_more_similar
        }, f, indent=2)

    # Calculate consistency metrics    
    metrics_map = {
        'npmi': 'Distinctiveness',
        'coherence': 'Semantic Coherence',
        'diversity': 'Topic Diversity',
        'kld': 'KL Divergence',
        'jsd': 'Jensen-Shannon Distance',
        'irbo': 'Semantic Integration',
        'overall_score': 'Overall Score'
    }
    
    consistency_data = []
    for metric in metrics_map.keys():
        values = [
            eval_distinct[metric],
            eval_similar[metric],
            eval_more_similar[metric]
        ]
        mean = np.mean(values)
        cv = (np.std(values) / mean) * 100  # Coefficient of Variation in percentage
        
        consistency_data.append({
            'Metric': metrics_map[metric],
            'Mean': mean,
            'CV (%)': cv
        })
    
    consistency_df = pd.DataFrame(consistency_data)
    consistency_df.set_index('Metric', inplace=True)
    
    print("\nOverall Consistency Analysis (Averaged across datasets)")
    print("=" * 60)
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    print(consistency_df.to_string())

if __name__ == '__main__':
    freeze_support()  # Required for multiprocessing support on Windows
    main()
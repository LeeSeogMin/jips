# DL_Eval.py
import numpy as np
import json
import pandas as pd
import pickle
from enhanced_evaluator import EnhancedTopicModelNeuralEvaluator

def calculate_statistics(results_list):
    """
    Calculate mean and CV for multiple evaluation runs
    """
    results_array = np.array(results_list)
    mean_values = np.mean(results_array, axis=0)
    std_values = np.std(results_array, axis=0)
    cv_values = (std_values / mean_values) * 100  # CV as percentage
    
    return mean_values, cv_values

def run_enhanced_evaluation(n_runs=5):
    # 1. Load preprocessed data
    df_distinct = pd.read_csv('data/distinct_topic.csv')
    df_similar = pd.read_csv('data/similar_topic.csv')
    df_more_similar = pd.read_csv('data/more_similar_topic.csv')

    # 2. Load precomputed embeddings and topics
    datasets = ['distinct', 'similar', 'more_similar']
    data_dict = {}
    
    for dataset in datasets:
        with open(f'data/embeddings_{dataset}.pkl', 'rb') as f:
            data_dict[f'embeddings_{dataset}'] = pickle.load(f)
        with open(f'data/topics_{dataset}.pkl', 'rb') as f:
            data_dict[f'topics_{dataset}'] = pickle.load(f)

    # 3. Initialize the enhanced evaluator
    evaluator = EnhancedTopicModelNeuralEvaluator()

    # 4. Convert labels to numeric
    def convert_labels_to_numeric(df):
        unique_labels = df['label'].unique()
        label_to_num = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        return df['label'].map(label_to_num).values

    # 5. Prepare evaluation data
    evaluation_data = {
        'distinct': {
            'df': df_distinct,
            'topics': data_dict['topics_distinct'],
            'assignments': convert_labels_to_numeric(df_distinct)
        },
        'similar': {
            'df': df_similar,
            'topics': data_dict['topics_similar'],
            'assignments': convert_labels_to_numeric(df_similar)
        },
        'more_similar': {
            'df': df_more_similar,
            'topics': data_dict['topics_more_similar'],
            'assignments': convert_labels_to_numeric(df_more_similar)
        }
    }

    # 6. Perform multiple evaluations
    metrics = [
        'Coherence',
        'Distinctiveness', 
        'Diversity',
        'Semantic Integration Score',
        'Overall Score'
    ]
    
    # 메트릭 표시 이름
    metric_display_names = [
        'Coherence',
        'Distinctiveness',
        'Diversity',
        'Semantic Integration',
        'Overall Score'
    ]
    
    all_results = {dataset_name: [] for dataset_name in evaluation_data.keys()}
    
    print("\nRunning evaluation with", n_runs, "iterations for consistency analysis...")
    
    for run in range(n_runs):
        print(f"Iteration {run + 1}/{n_runs}...")
        for dataset_name, dataset in evaluation_data.items():
            results = evaluator.evaluate(
                topics=dataset['topics'],
                docs=dataset['df']['text'].tolist(),
                topic_assignments=dataset['assignments']
            )
            all_results[dataset_name].append([results[m] for m in metrics])

    # 7. Calculate statistics and create tables
    dataset_tables = {}
    
    for dataset_name in evaluation_data.keys():
        mean_values, cv_values = calculate_statistics(all_results[dataset_name])
        
        # Create DataFrame with mean and CV columns
        dataset_stats = pd.DataFrame({
            'Mean': mean_values,
            'CV (%)': cv_values
        }, index=metric_display_names)
        
        dataset_tables[dataset_name] = dataset_stats

    # 8. Display topic keywords for each dataset
    print("\nTopic Keywords for Each Dataset")
    print("="*100)
    
    for dataset_name, dataset in evaluation_data.items():
        print(f"\n{dataset_name.replace('_', ' ').title()} Topics Keywords")
        print("-"*100)
        
        # Create DataFrame for topic keywords
        topic_df = pd.DataFrame({
            f'Topic {i+1}': keywords 
            for i, keywords in enumerate(dataset['topics'])
        }).T  # Transpose to make topics as rows
        
        # If topics have different numbers of keywords, pad with NaN
        max_keywords = max(len(keywords) for keywords in dataset['topics'])
        topic_df.columns = [f'Keyword {i+1}' for i in range(max_keywords)]
        
        print(topic_df.to_string())
        
        # Save topic keywords
        topic_df.to_csv(f'topic_keywords_{dataset_name}.csv')

    # 8. Display and save results
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    print("\nEvaluation Results for Individual Datasets")
    print("="*60)
    
    # Print tables for each dataset
    for dataset_name, table in dataset_tables.items():
        print(f"\n{dataset_name.replace('_', ' ').title()} Topics")
        print("-"*60)
        print(table.to_string())
        table.to_csv(f'evaluation_results_{dataset_name}.csv')
    
    # 9. Create overall consistency table (average metrics and CV across datasets)
    overall_means = np.mean([table['Mean'].values for table in dataset_tables.values()], axis=0)
    overall_cvs = np.mean([table['CV (%)'].values for table in dataset_tables.values()], axis=0)
    
    consistency_summary = pd.DataFrame({
        'Mean': overall_means,
        'CV (%)': overall_cvs
    }, index=metric_display_names)
    
    print("\nOverall Consistency Analysis (Averaged across datasets)")
    print("="*60)
    print(consistency_summary.to_string())
    consistency_summary.to_csv('overall_consistency.csv')
    
    # 10. Create dataset comparison table (average metrics across runs)
    comparison_data = {
        dataset_name.replace('_', ' ').title(): table['Mean'].values
        for dataset_name, table in dataset_tables.items()
    }
    
    comparison_table = pd.DataFrame(
        comparison_data,
        index=metric_display_names
    )
    
    print("\nDataset Comparison (Average Metrics)")
    print("="*60)
    print(comparison_table.to_string())
    comparison_table.to_csv('dataset_comparison.csv')
    
    # Save detailed results as JSON
    with open('detailed_evaluation_results.json', 'w') as f:
        json.dump({
            'all_runs': all_results,
            'statistics': {
                dataset: {
                    'mean': table['Mean'].tolist(),
                    'cv': table['CV (%)'].tolist()
                }
                for dataset, table in dataset_tables.items()
            },
            'overall_consistency': {
                'mean': consistency_summary['Mean'].tolist(),
                'average_cv': consistency_summary['CV (%)'].tolist()
            },
            'dataset_comparison': {
                dataset: values.tolist()
                for dataset, values in comparison_table.items()
            },
            'metadata': {
                'n_runs': n_runs,
                'metrics': dict(zip(metrics, metric_display_names))
            }
        }, f, indent=2, default=str)

    return dataset_tables, consistency_summary, comparison_table

if __name__ == "__main__":
    run_enhanced_evaluation(n_runs=5)
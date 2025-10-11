"""
Calculate Fleiss' Kappa for 3-LLM agreement (OpenAI, Anthropic, Grok)
Fleiss' κ measures inter-rater reliability for categorical data with 3+ raters
"""

import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import json

def load_llm_results():
    """Load all LLM evaluation results"""
    with open('data/openai_evaluation_results.pkl', 'rb') as f:
        openai = pickle.load(f)

    with open('data/anthropic_evaluation_results.pkl', 'rb') as f:
        anthropic = pickle.load(f)

    with open('data/grok_evaluation_results.pkl', 'rb') as f:
        grok = pickle.load(f)

    return openai, anthropic, grok

def categorize_scores(scores, bins=[0, 0.4, 0.7, 1.0]):
    """
    Convert continuous scores to categorical ratings
    Bins: [0, 0.4) = Low, [0.4, 0.7) = Medium, [0.7, 1.0] = High
    """
    categories = []
    for score in scores:
        if score < bins[1]:
            categories.append(0)  # Low
        elif score < bins[2]:
            categories.append(1)  # Medium
        else:
            categories.append(2)  # High
    return np.array(categories)

def calculate_fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for multiple raters

    Parameters:
    - ratings_matrix: (n_items, n_categories) matrix where each cell contains
                      the number of raters who assigned that category to that item

    Returns:
    - kappa: Fleiss' Kappa score
    """
    n_items, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Total raters per item (should be constant)

    # P_i: proportion of all assignments which were to the j-th category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # P_e: proportion of agreement expected by chance
    P_e = (p_j ** 2).sum()

    # P_i: extent to which raters agree for the i-th subject
    P_i = (ratings_matrix ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))

    # P_bar: mean of P_i
    P_bar = P_i.mean()

    # Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e)

    return kappa

def create_ratings_matrix(openai_cats, anthropic_cats, grok_cats, n_categories=3):
    """
    Create ratings matrix for Fleiss' Kappa

    Parameters:
    - openai_cats, anthropic_cats, grok_cats: categorical ratings from each LLM
    - n_categories: number of categories (3: Low, Medium, High)

    Returns:
    - ratings_matrix: (n_items, n_categories) matrix
    """
    n_items = len(openai_cats)
    ratings_matrix = np.zeros((n_items, n_categories), dtype=int)

    for i in range(n_items):
        # Count how many raters assigned each category to item i
        ratings_matrix[i, openai_cats[i]] += 1
        ratings_matrix[i, anthropic_cats[i]] += 1
        ratings_matrix[i, grok_cats[i]] += 1

    return ratings_matrix

def calculate_pairwise_cohen_kappa(openai_cats, anthropic_cats, grok_cats):
    """Calculate Cohen's Kappa for all pairs of raters"""
    kappas = {}

    # OpenAI vs Anthropic
    kappas['OpenAI-Anthropic'] = cohen_kappa_score(openai_cats, anthropic_cats)

    # OpenAI vs Grok
    kappas['OpenAI-Grok'] = cohen_kappa_score(openai_cats, grok_cats)

    # Anthropic vs Grok
    kappas['Anthropic-Grok'] = cohen_kappa_score(anthropic_cats, grok_cats)

    # Average Cohen's Kappa
    kappas['Average'] = np.mean([kappas['OpenAI-Anthropic'],
                                  kappas['OpenAI-Grok'],
                                  kappas['Anthropic-Grok']])

    return kappas

def main():
    print("=" * 60)
    print("Fleiss' Kappa Calculation for 3-LLM Agreement")
    print("=" * 60)

    # Load LLM results
    openai, anthropic, grok = load_llm_results()

    datasets = ['Distinct Topics', 'Similar Topics', 'More Similar Topics']
    metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']

    results = {}

    for metric in metrics:
        print(f"\n{'=' * 60}")
        print(f"Metric: {metric.upper()}")
        print('=' * 60)

        # Extract scores for this metric across all datasets
        openai_scores = []
        anthropic_scores = []
        grok_scores = []

        for dataset in datasets:
            openai_scores.append(openai[dataset]['scores'][metric])
            anthropic_scores.append(anthropic[dataset]['scores'][metric])
            grok_scores.append(grok[dataset]['scores'][metric])

        # Convert to numpy arrays
        openai_scores = np.array(openai_scores)
        anthropic_scores = np.array(anthropic_scores)
        grok_scores = np.array(grok_scores)

        # Categorize scores (continuous -> categorical)
        openai_cats = categorize_scores(openai_scores)
        anthropic_cats = categorize_scores(anthropic_scores)
        grok_cats = categorize_scores(grok_scores)

        print(f"\nContinuous Scores:")
        print(f"  OpenAI:    {openai_scores}")
        print(f"  Anthropic: {anthropic_scores}")
        print(f"  Grok:      {grok_scores}")

        print(f"\nCategorical Ratings (0=Low, 1=Medium, 2=High):")
        print(f"  OpenAI:    {openai_cats}")
        print(f"  Anthropic: {anthropic_cats}")
        print(f"  Grok:      {grok_cats}")

        # Create ratings matrix
        ratings_matrix = create_ratings_matrix(openai_cats, anthropic_cats, grok_cats)
        print(f"\nRatings Matrix (rows=items, cols=categories):")
        print(f"  {ratings_matrix}")

        # Calculate Fleiss' Kappa
        fleiss_kappa = calculate_fleiss_kappa(ratings_matrix)
        print(f"\nFleiss' Kappa: {fleiss_kappa:.3f}")

        # Interpretation
        if fleiss_kappa < 0:
            interpretation = "Poor (worse than chance)"
        elif fleiss_kappa < 0.20:
            interpretation = "Slight"
        elif fleiss_kappa < 0.40:
            interpretation = "Fair"
        elif fleiss_kappa < 0.60:
            interpretation = "Moderate"
        elif fleiss_kappa < 0.80:
            interpretation = "Substantial"
        else:
            interpretation = "Almost Perfect"

        print(f"Interpretation: {interpretation}")

        # Calculate pairwise Cohen's Kappa
        cohen_kappas = calculate_pairwise_cohen_kappa(openai_cats, anthropic_cats, grok_cats)
        print(f"\nPairwise Cohen's Kappa:")
        for pair, kappa in cohen_kappas.items():
            print(f"  {pair}: {kappa:.3f}")

        # Store results
        results[metric] = {
            'fleiss_kappa': float(fleiss_kappa),
            'interpretation': interpretation,
            'cohen_kappas': {k: float(v) for k, v in cohen_kappas.items()},
            'ratings_matrix': ratings_matrix.tolist(),
            'categorical_ratings': {
                'OpenAI': openai_cats.tolist(),
                'Anthropic': anthropic_cats.tolist(),
                'Grok': grok_cats.tolist()
            }
        }

    # Calculate overall agreement across all metrics
    print(f"\n{'=' * 60}")
    print("Overall Agreement Across All Metrics")
    print('=' * 60)

    all_fleiss_kappas = [results[m]['fleiss_kappa'] for m in metrics]
    mean_fleiss = np.mean(all_fleiss_kappas)
    std_fleiss = np.std(all_fleiss_kappas)

    print(f"\nFleiss' Kappa by Metric:")
    for metric in metrics:
        print(f"  {metric:25s}: {results[metric]['fleiss_kappa']:.3f} ({results[metric]['interpretation']})")

    print(f"\nMean Fleiss' Kappa: {mean_fleiss:.3f} ± {std_fleiss:.3f}")

    # Overall interpretation
    if mean_fleiss < 0.40:
        overall_interp = "Fair to Moderate Agreement"
    elif mean_fleiss < 0.60:
        overall_interp = "Moderate Agreement"
    elif mean_fleiss < 0.80:
        overall_interp = "Substantial Agreement"
    else:
        overall_interp = "Almost Perfect Agreement"

    print(f"Overall Interpretation: {overall_interp}")

    # Add summary statistics
    results['summary'] = {
        'mean_fleiss_kappa': float(mean_fleiss),
        'std_fleiss_kappa': float(std_fleiss),
        'interpretation': overall_interp,
        'n_raters': 3,
        'n_items': 3,
        'n_metrics': 4,
        'categorization_bins': [0, 0.4, 0.7, 1.0],
        'category_labels': ['Low (0-0.4)', 'Medium (0.4-0.7)', 'High (0.7-1.0)']
    }

    # Save results to JSON
    output_file = 'data/llm_agreement_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

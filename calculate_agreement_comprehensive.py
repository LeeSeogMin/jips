"""
Comprehensive Inter-rater Agreement Analysis for 3-LLM Evaluation
Uses all 12 data points (4 metrics × 3 datasets) for robust statistics
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

def categorize_scores(scores, bins=[0, 0.5, 0.75, 1.0]):
    """
    Convert continuous scores to categorical ratings
    Bins: [0, 0.5) = Low, [0.5, 0.75) = Medium, [0.75, 1.0] = High
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

    # Check if all items have same number of raters
    if not all(ratings_matrix.sum(axis=1) == n_raters):
        raise ValueError("All items must have the same number of raters")

    # P_j: proportion of all assignments which were to the j-th category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # P_e: proportion of agreement expected by chance
    P_e = (p_j ** 2).sum()

    # P_i: extent to which raters agree for the i-th subject
    P_i = (ratings_matrix ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))

    # P_bar: mean of P_i
    P_bar = P_i.mean()

    # Fleiss' Kappa
    if P_e == 1.0:  # Perfect agreement by chance (all same category)
        return np.nan
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

    try:
        kappas['OpenAI-Anthropic'] = cohen_kappa_score(openai_cats, anthropic_cats)
    except:
        kappas['OpenAI-Anthropic'] = np.nan

    try:
        kappas['OpenAI-Grok'] = cohen_kappa_score(openai_cats, grok_cats)
    except:
        kappas['OpenAI-Grok'] = np.nan

    try:
        kappas['Anthropic-Grok'] = cohen_kappa_score(anthropic_cats, grok_cats)
    except:
        kappas['Anthropic-Grok'] = np.nan

    # Average Cohen's Kappa (ignore NaN)
    valid_kappas = [k for k in kappas.values() if not np.isnan(k)]
    kappas['Average'] = np.mean(valid_kappas) if valid_kappas else np.nan

    return kappas

def main():
    print("=" * 70)
    print("Comprehensive Inter-rater Agreement Analysis (3 LLMs)")
    print("=" * 70)

    # Load LLM results
    openai, anthropic, grok = load_llm_results()

    datasets = ['Distinct Topics', 'Similar Topics', 'More Similar Topics']
    metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']

    # Collect ALL scores (4 metrics × 3 datasets = 12 data points)
    all_openai_scores = []
    all_anthropic_scores = []
    all_grok_scores = []
    score_labels = []

    for metric in metrics:
        for dataset in datasets:
            all_openai_scores.append(openai[dataset]['scores'][metric])
            all_anthropic_scores.append(anthropic[dataset]['scores'][metric])
            all_grok_scores.append(grok[dataset]['scores'][metric])
            score_labels.append(f"{metric[:4]}_{dataset[:4]}")

    # Convert to numpy arrays
    all_openai_scores = np.array(all_openai_scores)
    all_anthropic_scores = np.array(all_anthropic_scores)
    all_grok_scores = np.array(all_grok_scores)

    print(f"\nTotal data points: {len(all_openai_scores)} (4 metrics × 3 datasets)")
    print(f"Score range: [{all_openai_scores.min():.3f}, {all_openai_scores.max():.3f}]")

    # === Method 1: Fleiss' Kappa on ALL 12 data points ===
    print(f"\n{'=' * 70}")
    print("Method 1: Fleiss' Kappa (Categorical Agreement)")
    print('=' * 70)

    # Categorize scores
    openai_cats = categorize_scores(all_openai_scores)
    anthropic_cats = categorize_scores(all_anthropic_scores)
    grok_cats = categorize_scores(all_grok_scores)

    print(f"\nCategorization bins: [0, 0.5) = Low, [0.5, 0.75) = Medium, [0.75, 1.0] = High")
    print(f"\nCategory distribution:")
    print(f"  OpenAI:    Low={np.sum(openai_cats == 0)}, Med={np.sum(openai_cats == 1)}, High={np.sum(openai_cats == 2)}")
    print(f"  Anthropic: Low={np.sum(anthropic_cats == 0)}, Med={np.sum(anthropic_cats == 1)}, High={np.sum(anthropic_cats == 2)}")
    print(f"  Grok:      Low={np.sum(grok_cats == 0)}, Med={np.sum(grok_cats == 1)}, High={np.sum(grok_cats == 2)}")

    # Create ratings matrix
    ratings_matrix = create_ratings_matrix(openai_cats, anthropic_cats, grok_cats)

    # Calculate Fleiss' Kappa
    fleiss_kappa = calculate_fleiss_kappa(ratings_matrix)

    if np.isnan(fleiss_kappa):
        print(f"\nFleiss' Kappa: N/A (perfect agreement or no variance)")
        fleiss_interp = "N/A"
    else:
        print(f"\nFleiss' Kappa: {fleiss_kappa:.3f}")

        # Interpretation
        if fleiss_kappa < 0:
            fleiss_interp = "Poor (worse than chance)"
        elif fleiss_kappa < 0.20:
            fleiss_interp = "Slight"
        elif fleiss_kappa < 0.40:
            fleiss_interp = "Fair"
        elif fleiss_kappa < 0.60:
            fleiss_interp = "Moderate"
        elif fleiss_kappa < 0.80:
            fleiss_interp = "Substantial"
        else:
            fleiss_interp = "Almost Perfect"

        print(f"Interpretation: {fleiss_interp}")

    # Calculate pairwise Cohen's Kappa
    cohen_kappas = calculate_pairwise_cohen_kappa(openai_cats, anthropic_cats, grok_cats)
    print(f"\nPairwise Cohen's Kappa:")
    for pair, kappa in cohen_kappas.items():
        if np.isnan(kappa):
            print(f"  {pair}: N/A")
        else:
            print(f"  {pair}: {kappa:.3f}")

    # === Method 2: Pearson Correlation (already computed in previous analysis) ===
    print(f"\n{'=' * 70}")
    print("Method 2: Pearson Correlation (Continuous Score Agreement)")
    print('=' * 70)

    pearson_results = {}

    # OpenAI vs Anthropic
    r_oa, p_oa = pearsonr(all_openai_scores, all_anthropic_scores)
    pearson_results['OpenAI-Anthropic'] = {'r': r_oa, 'p': p_oa}
    print(f"\nOpenAI vs Anthropic:  r = {r_oa:.3f} (p = {p_oa:.4f})")

    # OpenAI vs Grok
    r_og, p_og = pearsonr(all_openai_scores, all_grok_scores)
    pearson_results['OpenAI-Grok'] = {'r': r_og, 'p': p_og}
    print(f"OpenAI vs Grok:       r = {r_og:.3f} (p = {p_og:.4f})")

    # Anthropic vs Grok
    r_ag, p_ag = pearsonr(all_anthropic_scores, all_grok_scores)
    pearson_results['Anthropic-Grok'] = {'r': r_ag, 'p': p_ag}
    print(f"Anthropic vs Grok:    r = {r_ag:.3f} (p = {p_ag:.4f})")

    # Average
    avg_r = np.mean([r_oa, r_og, r_ag])
    pearson_results['Average'] = {'r': avg_r}
    print(f"\nAverage Pearson r:    {avg_r:.3f}")

    # Interpretation
    if avg_r >= 0.9:
        pearson_interp = "Very Strong Agreement"
    elif avg_r >= 0.7:
        pearson_interp = "Strong Agreement"
    elif avg_r >= 0.5:
        pearson_interp = "Moderate Agreement"
    elif avg_r >= 0.3:
        pearson_interp = "Weak Agreement"
    else:
        pearson_interp = "Very Weak Agreement"

    print(f"Interpretation: {pearson_interp}")

    # === Method 3: Mean Absolute Error ===
    print(f"\n{'=' * 70}")
    print("Method 3: Mean Absolute Error (MAE)")
    print('=' * 70)

    mae_results = {}

    # OpenAI vs Anthropic
    mae_oa = np.mean(np.abs(all_openai_scores - all_anthropic_scores))
    mae_results['OpenAI-Anthropic'] = mae_oa
    print(f"\nOpenAI vs Anthropic:  MAE = {mae_oa:.3f}")

    # OpenAI vs Grok
    mae_og = np.mean(np.abs(all_openai_scores - all_grok_scores))
    mae_results['OpenAI-Grok'] = mae_og
    print(f"OpenAI vs Grok:       MAE = {mae_og:.3f}")

    # Anthropic vs Grok
    mae_ag = np.mean(np.abs(all_anthropic_scores - all_grok_scores))
    mae_results['Anthropic-Grok'] = mae_ag
    print(f"Anthropic vs Grok:    MAE = {mae_ag:.3f}")

    # Average
    avg_mae = np.mean([mae_oa, mae_og, mae_ag])
    mae_results['Average'] = avg_mae
    print(f"\nAverage MAE:          {avg_mae:.3f}")

    # Interpretation
    if avg_mae < 0.05:
        mae_interp = "Excellent (very low disagreement)"
    elif avg_mae < 0.10:
        mae_interp = "Good (low disagreement)"
    elif avg_mae < 0.15:
        mae_interp = "Fair (moderate disagreement)"
    else:
        mae_interp = "Poor (high disagreement)"

    print(f"Interpretation: {mae_interp}")

    # === Summary ===
    print(f"\n{'=' * 70}")
    print("Summary: 3-LLM Agreement Across All Metrics")
    print('=' * 70)

    fleiss_str = f"{fleiss_kappa:.3f}" if not np.isnan(fleiss_kappa) else "N/A"
    print(f"\n1. Fleiss' Kappa (categorical):   {fleiss_str} ({fleiss_interp})")
    print(f"2. Pearson Correlation (continuous): {avg_r:.3f} ({pearson_interp})")
    print(f"3. Mean Absolute Error:           {avg_mae:.3f} ({mae_interp})")

    print(f"\n{'=' * 70}")
    print("Conclusion:")
    print('=' * 70)

    if avg_r >= 0.85:
        conclusion = "✅ EXCELLENT inter-rater reliability across all 3 LLMs"
    elif avg_r >= 0.7:
        conclusion = "✅ STRONG inter-rater reliability across all 3 LLMs"
    elif avg_r >= 0.5:
        conclusion = "⚠️ MODERATE inter-rater reliability across all 3 LLMs"
    else:
        conclusion = "❌ WEAK inter-rater reliability across all 3 LLMs"

    print(conclusion)

    # Save comprehensive results
    results = {
        'fleiss_kappa': {
            'value': float(fleiss_kappa) if not np.isnan(fleiss_kappa) else None,
            'interpretation': fleiss_interp,
            'n_items': len(all_openai_scores),
            'n_raters': 3,
            'categorization_bins': [0, 0.5, 0.75, 1.0],
            'category_labels': ['Low (0-0.5)', 'Medium (0.5-0.75)', 'High (0.75-1.0)'],
            'cohen_kappas': {k: float(v) if not np.isnan(v) else None for k, v in cohen_kappas.items()}
        },
        'pearson_correlation': {
            'pairwise': {k: {'r': float(v['r']), 'p': float(v['p'])} if 'p' in v else {'r': float(v['r'])}
                        for k, v in pearson_results.items()},
            'average_r': float(avg_r),
            'interpretation': pearson_interp
        },
        'mean_absolute_error': {
            'pairwise': {k: float(v) for k, v in mae_results.items()},
            'interpretation': mae_interp
        },
        'summary': {
            'n_llms': 3,
            'n_datasets': 3,
            'n_metrics': 4,
            'total_data_points': len(all_openai_scores),
            'conclusion': conclusion
        }
    }

    output_file = 'data/llm_agreement_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Comprehensive results saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

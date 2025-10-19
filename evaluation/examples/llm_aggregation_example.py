"""
LLM Score Aggregation Example

This example demonstrates the weighted ensemble aggregation of LLM evaluation
scores and the calculation of rank-ordering consistency using Spearman correlation.

Weighted ensemble: 0.35×Claude + 0.40×GPT + 0.25×Grok

Reference: "Semantic-based Evaluation Framework for Topic Models"
"""

import numpy as np
from scipy.stats import spearmanr


def aggregate_llm_scores(claude_scores, gpt_scores, grok_scores):
    """
    Aggregate LLM scores using weighted ensemble.
    
    Weights: 0.35×Claude + 0.40×GPT + 0.25×Grok
    
    Args:
        claude_scores, gpt_scores, grok_scores: list of float scores
    
    Returns:
        numpy array: aggregated scores
    """
    weights = {'claude': 0.35, 'gpt': 0.40, 'grok': 0.25}
    
    claude = np.array(claude_scores)
    gpt = np.array(gpt_scores)
    grok = np.array(grok_scores)
    
    ensemble = (weights['claude'] * claude + 
                weights['gpt'] * gpt + 
                weights['grok'] * grok)
    
    return ensemble


def calculate_model_agreement(scores1, scores2, name1="Model 1", name2="Model 2"):
    """
    Calculate Spearman correlation for rank-ordering consistency.
    
    Args:
        scores1, scores2: list or array of scores
        name1, name2: model names for display
    
    Returns:
        tuple: (correlation coefficient, p-value)
    """
    rho, pval = spearmanr(scores1, scores2)
    print(f"  {name1} vs {name2}: ρ = {rho:.3f} (p = {pval:.4f})")
    return rho, pval


def main():
    """Run the LLM score aggregation example."""
    
    print("="*70)
    print("LLM SCORE AGGREGATION")
    print("="*70)
    
    # Example: Three LLMs evaluate three topics
    print("\nExample: Three LLMs evaluate three topics with coherence scores")
    print("\n| Topic | Claude-sonnet-4-5 | GPT-4.1 | Grok-4 |")
    print("|-------|-------------------|---------|--------|")
    
    claude = [0.920, 0.820, 0.780]
    gpt = [0.920, 0.920, 0.890]
    grok = [0.950, 0.950, 0.920]
    
    for i, (c, g, k) in enumerate(zip(claude, gpt, grok), 1):
        print(f"| {i}     | {c:.3f}             | {g:.3f}   | {k:.3f}  |")
    
    # Weighted ensemble aggregation
    print("\n" + "="*70)
    print("WEIGHTED ENSEMBLE AGGREGATION")
    print("="*70)
    print("\nWeights: 0.35×Claude + 0.40×GPT + 0.25×Grok")
    print("\nCalculation:")
    
    ensemble = aggregate_llm_scores(claude, gpt, grok)
    
    for i, (c, g, k, e) in enumerate(zip(claude, gpt, grok, ensemble), 1):
        calc = f"0.35×{c:.3f} + 0.40×{g:.3f} + 0.25×{k:.3f}"
        print(f"  Topic {i}: {calc} = {e:.3f}")
    
    print(f"\nEnsemble scores: {ensemble}")
    print(f"Expected: [0.928, 0.892, 0.859]")
    
    # Pairwise agreement (Spearman correlation)
    print("\n" + "="*70)
    print("RANK-ORDERING CONSISTENCY (Spearman ρ)")
    print("="*70)
    print("\nPairwise correlations:")
    
    rho_cg, _ = calculate_model_agreement(claude, grok, "Claude", "Grok")
    rho_gg, _ = calculate_model_agreement(gpt, grok, "GPT", "Grok")
    rho_cp, _ = calculate_model_agreement(claude, gpt, "Claude", "GPT")
    
    mean_rho = (rho_cg + rho_gg + rho_cp) / 3
    print(f"\nMean pairwise correlation: ρ = {mean_rho:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Ensemble scores: {ensemble}")
    print(f"Mean correlation: ρ = {mean_rho:.3f}")
    print(f"\nInterpretation: The weighted ensemble combines the strengths of")
    print("three LLMs, with GPT-4.1 weighted highest (0.40) for balanced")
    print("assessment, Claude-sonnet-4-5 (0.35) for conservative perspective,")
    print("and Grok-4 (0.25) weighted lower due to lenient tendency.")
    print(f"\nHigh pairwise correlations (mean ρ = {mean_rho:.3f}) indicate")
    print("strong rank-ordering consistency across models.")
    print("="*70)
    
    # Note on Cohen's κ
    print("\n" + "="*70)
    print("NOTE ON COHEN'S κ")
    print("="*70)
    print("We use Spearman correlation (ρ) rather than Cohen's κ or Fleiss' κ")
    print("because our LLM evaluations produce continuous scores (0-1 scale),")
    print("not categorical labels. Kappa metrics require categorical data.")
    print("Spearman ρ directly measures rank-ordering consistency across models.")
    print("="*70)


if __name__ == "__main__":
    main()


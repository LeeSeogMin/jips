"""
Semantic Diversity Calculation Example

This example demonstrates the calculation of Semantic Diversity (SemDiv) using
the formula from Section 3.3.2 of the manuscript:

    SemDiv = (D_semantic + D_distribution) / 2

where:
- D_semantic is the mean pairwise distinctiveness
- D_distribution is the normalized entropy of topic distribution

Reference: "Semantic-based Evaluation Framework for Topic Models"
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


def calculate_semantic_diversity(topic_embeddings, topic_assignments):
    """
    Calculate semantic diversity combining semantic and distribution diversity.
    
    Formula: SemDiv = (D_semantic + D_distribution) / 2
    where D_semantic is mean pairwise distinctiveness,
    and D_distribution is normalized entropy
    
    Args:
        topic_embeddings: numpy array (n_topics, embedding_dim)
        topic_assignments: list of topic indices for documents
    
    Returns:
        tuple: (diversity, semantic_diversity, distribution_diversity)
    """
    print("="*70)
    print("SEMANTIC DIVERSITY CALCULATION")
    print("="*70)
    
    n_topics = len(topic_embeddings)
    
    # Step 1: Calculate semantic diversity (mean pairwise distinctiveness)
    print("\nStep 1: Calculate semantic diversity (pairwise distinctiveness)")
    pairwise_dist = []
    for i in range(n_topics):
        for j in range(i+1, n_topics):
            sim = cosine_similarity(
                topic_embeddings[i].reshape(1, -1),
                topic_embeddings[j].reshape(1, -1)
            )[0, 0]
            distinctiveness = (1 - sim) / 2
            pairwise_dist.append(distinctiveness)
            print(f"  cos(t_{i}, t_{j}) = {sim:.3f} → SD({i},{j}) = {distinctiveness:.3f}")
    
    semantic_diversity = np.mean(pairwise_dist)
    print(f"\nD_semantic = mean of pairwise distinctiveness = {semantic_diversity:.3f}")
    
    # Step 2: Calculate distribution diversity (normalized entropy)
    print("\nStep 2: Calculate distribution diversity (normalized entropy)")
    topic_counts = np.bincount(topic_assignments, minlength=n_topics)
    topic_probs = topic_counts / len(topic_assignments)
    
    print(f"Document distribution:")
    for i, (count, prob) in enumerate(zip(topic_counts, topic_probs)):
        print(f"  Topic {i}: {count} documents → p_{i} = {prob:.3f}")
    
    H_T = entropy(topic_probs)
    H_max = np.log(n_topics)
    distribution_diversity = H_T / H_max if H_max > 0 else 0.0
    
    print(f"\nH_T (entropy) = {H_T:.3f}")
    print(f"H_max = log({n_topics}) = {H_max:.3f}")
    print(f"D_distribution = H_T / H_max = {distribution_diversity:.3f}")
    
    # Step 3: Calculate combined diversity
    print("\nStep 3: Calculate combined diversity")
    print("Formula: SemDiv = (D_semantic + D_distribution) / 2")
    diversity = (semantic_diversity + distribution_diversity) / 2
    
    print(f"SemDiv = ({semantic_diversity:.3f} + {distribution_diversity:.3f}) / 2 = {diversity:.3f}")
    
    return diversity, semantic_diversity, distribution_diversity


def main():
    """Run the semantic diversity example."""
    
    print("\nToy Example: Three topics with document assignments")
    print("\nTopic centroids (simplified 3D):")
    
    # Topic centroids
    topic_centroids = np.array([[0.80, 0.40, 0.20],  # ML
                                [0.20, 0.80, 0.60],  # Automotive
                                [0.30, 0.50, 0.85]]) # Biology
    
    topic_names = ["Machine Learning", "Automotive", "Biology"]
    for i, (name, centroid) in enumerate(zip(topic_names, topic_centroids)):
        print(f"  Topic {i} ({name}): {centroid}")
    
    # Document assignments (4 docs to topic 0, 3 to topic 1, 5 to topic 2)
    print("\nDocument assignments:")
    assignments = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    print(f"  {assignments}")
    print(f"  Total: {len(assignments)} documents")
    
    # Calculate semantic diversity
    semdiv, d_sem, d_dist = calculate_semantic_diversity(topic_centroids, assignments)
    
    # Summary
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Semantic Diversity: {semdiv:.3f}")
    print(f"  D_semantic:      {d_sem:.3f}")
    print(f"  D_distribution:  {d_dist:.3f}")
    print(f"Expected: ~0.559")
    print(f"\nInterpretation: Diversity score ({semdiv:.3f}) indicates moderate")
    print("topic separation in semantic space combined with near-uniform")
    print("document distribution, suggesting a reasonably diverse topic model.")
    print("="*70)


if __name__ == "__main__":
    main()


"""
Semantic Coherence Calculation Example

This example demonstrates the calculation of Semantic Coherence (SC) using
the formula from Section 3.3.2 of the manuscript:

    SC = Σ(w_ij · h_ij) / Σ(w_ij)

where:
- h_ij is the pairwise similarity matrix between keywords
- w_ij = w_i × w_j is the importance weight matrix from PageRank

Reference: "Semantic-based Evaluation Framework for Topic Models"
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def calculate_semantic_coherence(keyword_embeddings, keywords):
    """
    Calculate semantic coherence with PageRank weighting.
    
    Formula: SC = Σ(w_ij · h_ij) / Σ(w_ij)
    where h_ij is pairwise similarity and w_ij = w_i × w_j
    
    Args:
        keyword_embeddings: numpy array (n_keywords, embedding_dim)
        keywords: list of keyword strings
    
    Returns:
        float: coherence score in [0, 1]
    """
    print("="*70)
    print("SEMANTIC COHERENCE CALCULATION")
    print("="*70)
    
    # Step 1: Calculate pairwise similarity matrix (h_ij)
    print("\nStep 1: Calculate pairwise similarity matrix (h_ij)")
    similarity_matrix = cosine_similarity(keyword_embeddings)
    print("Similarity matrix:")
    print(similarity_matrix)
    print(f"h_01 = {similarity_matrix[0,1]:.3f}")
    print(f"h_02 = {similarity_matrix[0,2]:.3f}")
    print(f"h_12 = {similarity_matrix[1,2]:.3f}")
    
    # Step 2: Build semantic graph and calculate PageRank weights
    print("\nStep 2: Calculate PageRank weights from semantic graph")
    G = nx.from_numpy_array(similarity_matrix)
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    weights = np.array([pagerank_scores[i] for i in range(len(keywords))])
    print(f"PageRank weights:")
    for i, (keyword, weight) in enumerate(zip(keywords, weights)):
        print(f"  w_{keyword} = {weight:.3f}")
    
    # Step 3: Create importance weight matrix (w_ij = w_i × w_j)
    print("\nStep 3: Create importance weight matrix (w_ij = w_i × w_j)")
    weight_matrix = np.outer(weights, weights)
    print("Weight matrix:")
    print(weight_matrix)
    print(f"Σw_ij = {weight_matrix.sum():.3f}")
    
    # Step 4: Calculate weighted coherence
    print("\nStep 4: Calculate weighted coherence")
    print("Formula: SC = Σ(w_ij · h_ij) / Σ(w_ij)")
    numerator = np.sum(weight_matrix * similarity_matrix)
    denominator = np.sum(weight_matrix)
    coherence = numerator / denominator
    
    print(f"Numerator = Σ(w_ij · h_ij) = {numerator:.4f}")
    print(f"Denominator = Σ(w_ij) = {denominator:.4f}")
    print(f"SC = {numerator:.4f} / {denominator:.4f} = {coherence:.3f}")
    
    return coherence


def main():
    """Run the semantic coherence example."""
    
    # Toy example: Topic with three keywords
    print("\nToy Example: Topic = ['neural', 'network', 'learning']")
    print("\nEmbeddings (simplified 3D for illustration):")
    
    embeddings = np.array([[0.9, 0.3, 0.1],
                           [0.8, 0.4, 0.2],
                           [0.7, 0.5, 0.3]])
    keywords = ["neural", "network", "learning"]
    
    for keyword, emb in zip(keywords, embeddings):
        print(f"  e_{keyword} = {emb}")
    
    # Calculate semantic coherence
    sc = calculate_semantic_coherence(embeddings, keywords)
    
    # Summary
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Semantic Coherence: {sc:.3f}")
    print(f"Expected: ~0.980")
    print(f"\nInterpretation: High coherence score ({sc:.3f}) indicates strong")
    print("semantic relatedness among keywords 'neural', 'network', and 'learning',")
    print("which form a cohesive topic about neural networks.")
    print("="*70)


if __name__ == "__main__":
    main()


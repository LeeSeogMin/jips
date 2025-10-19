"""
Semantic Distinctiveness Calculation Example

This example demonstrates the calculation of Semantic Distinctiveness (SD) using
the formula from Section 3.3.2 of the manuscript:

    SD = (1 - cos(t_i, t_j)) / 2

where t_i and t_j are topic centroid embeddings (mean of keyword embeddings).

Reference: "Semantic-based Evaluation Framework for Topic Models"
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_semantic_distinctiveness(topic_keywords_1, topic_keywords_2, 
                                      keyword_embeddings_1, keyword_embeddings_2):
    """
    Calculate semantic distinctiveness between two topics.
    
    Formula: SD = (1 - cos(t_i, t_j)) / 2
    where t_i, t_j are topic centroid embeddings
    
    Args:
        topic_keywords_1, topic_keywords_2: lists of keyword strings
        keyword_embeddings_1, keyword_embeddings_2: numpy arrays (n_keywords, embedding_dim)
    
    Returns:
        float: distinctiveness score in [0, 1]
    """
    print("="*70)
    print("SEMANTIC DISTINCTIVENESS CALCULATION")
    print("="*70)
    
    # Step 1: Calculate topic centroids (mean of keyword embeddings)
    print("\nStep 1: Calculate topic centroids")
    topic_centroid_1 = keyword_embeddings_1.mean(axis=0)
    topic_centroid_2 = keyword_embeddings_2.mean(axis=0)
    
    print(f"Topic A keywords: {topic_keywords_1}")
    print(f"Topic A centroid: {topic_centroid_1}")
    print(f"\nTopic B keywords: {topic_keywords_2}")
    print(f"Topic B centroid: {topic_centroid_2}")
    
    # Step 2: Calculate cosine similarity between centroids
    print("\nStep 2: Calculate cosine similarity between centroids")
    similarity = cosine_similarity(
        topic_centroid_1.reshape(1, -1),
        topic_centroid_2.reshape(1, -1)
    )[0, 0]
    
    print(f"cos(t_A, t_B) = {similarity:.3f}")
    
    # Step 3: Transform to distinctiveness [0, 1] range
    print("\nStep 3: Transform to distinctiveness")
    print("Formula: SD = (1 - cos(t_A, t_B)) / 2")
    distinctiveness = (1 - similarity) / 2
    
    print(f"SD = (1 - {similarity:.3f}) / 2 = {distinctiveness:.3f}")
    
    return distinctiveness


def main():
    """Run the semantic distinctiveness example."""
    
    print("\nToy Example: Two topics from different domains")
    print("\nTopic A: Machine Learning")
    print("  Keywords: ['neural', 'network', 'learning']")
    print("  Embeddings (simplified 3D):")
    
    # Topic A: Machine learning
    embeddings_a = np.array([[0.9, 0.3, 0.1],
                             [0.8, 0.4, 0.2],
                             [0.7, 0.5, 0.3]])
    keywords_a = ["neural", "network", "learning"]
    
    for keyword, emb in zip(keywords_a, embeddings_a):
        print(f"    e_{keyword} = {emb}")
    
    print("\nTopic B: Automotive")
    print("  Keywords: ['engine', 'vehicle', 'motor']")
    print("  Embeddings (simplified 3D):")
    
    # Topic B: Automotive
    embeddings_b = np.array([[0.2, 0.8, 0.6],
                             [0.3, 0.7, 0.5],
                             [0.1, 0.9, 0.7]])
    keywords_b = ["engine", "vehicle", "motor"]
    
    for keyword, emb in zip(keywords_b, embeddings_b):
        print(f"    e_{keyword} = {emb}")
    
    # Calculate semantic distinctiveness
    sd = calculate_semantic_distinctiveness(keywords_a, keywords_b, 
                                           embeddings_a, embeddings_b)
    
    # Summary
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Semantic Distinctiveness: {sd:.3f}")
    print(f"Expected: ~0.171")
    print(f"\nInterpretation: Moderate distinctiveness score ({sd:.3f}) indicates")
    print("that while the topics are from different domains (ML vs automotive),")
    print("they share some semantic overlap in the embedding space.")
    print("Higher values would indicate more distinct topics.")
    print("="*70)


if __name__ == "__main__":
    main()


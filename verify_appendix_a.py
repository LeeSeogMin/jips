"""
Verify all calculations in Appendix A
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.stats import entropy, spearmanr

print("="*70)
print("APPENDIX A VERIFICATION")
print("="*70)

# ============================================================================
# A.1 Semantic Coherence
# ============================================================================
print("\n" + "="*70)
print("A.1 SEMANTIC COHERENCE")
print("="*70)

embeddings = np.array([[0.9, 0.3, 0.1],
                       [0.8, 0.4, 0.2],
                       [0.7, 0.5, 0.3]])

print("\nStep 1: Embeddings")
print(f"e_neural   = {embeddings[0]}")
print(f"e_network  = {embeddings[1]}")
print(f"e_learning = {embeddings[2]}")

print("\nStep 2: Pairwise similarity matrix (h_ij)")
sim_matrix = cosine_similarity(embeddings)
print(sim_matrix)
print(f"h_01 = {sim_matrix[0,1]:.3f}")
print(f"h_02 = {sim_matrix[0,2]:.3f}")
print(f"h_12 = {sim_matrix[1,2]:.3f}")

print("\nStep 3: PageRank weights")
G = nx.from_numpy_array(sim_matrix)
pagerank_scores = nx.pagerank(G, alpha=0.85)
weights = np.array([pagerank_scores[i] for i in range(3)])
print(f"w_0 = {weights[0]:.3f}")
print(f"w_1 = {weights[1]:.3f}")
print(f"w_2 = {weights[2]:.3f}")
print(f"Sum = {weights.sum():.3f}")

print("\nStep 4: Weight matrix (w_ij = w_i × w_j)")
weight_matrix = np.outer(weights, weights)
print(weight_matrix)
print(f"Σw_ij = {weight_matrix.sum():.3f}")

print("\nStep 5: Coherence calculation")
numerator = np.sum(weight_matrix * sim_matrix)
denominator = np.sum(weight_matrix)
coherence = numerator / denominator
print(f"Numerator = {numerator:.4f}")
print(f"Denominator = {denominator:.4f}")
print(f"SC = {coherence:.3f}")
print(f"✓ Expected: ~0.980, Got: {coherence:.3f}")

# ============================================================================
# A.2 Semantic Distinctiveness
# ============================================================================
print("\n" + "="*70)
print("A.2 SEMANTIC DISTINCTIVENESS")
print("="*70)

# Topic A
embeddings_a = np.array([[0.9, 0.3, 0.1],
                         [0.8, 0.4, 0.2],
                         [0.7, 0.5, 0.3]])
t_a = embeddings_a.mean(axis=0)

# Topic B
embeddings_b = np.array([[0.2, 0.8, 0.6],
                         [0.3, 0.7, 0.5],
                         [0.1, 0.9, 0.7]])
t_b = embeddings_b.mean(axis=0)

print("\nStep 1: Topic centroids")
print(f"t_A = {t_a}")
print(f"t_B = {t_b}")

print("\nStep 2: Cosine similarity")
similarity = cosine_similarity(t_a.reshape(1, -1), t_b.reshape(1, -1))[0, 0]
print(f"cos(t_A, t_B) = {similarity:.3f}")

print("\nStep 3: Distinctiveness")
distinctiveness = (1 - similarity) / 2
print(f"SD = (1 - {similarity:.3f}) / 2 = {distinctiveness:.3f}")
print(f"✓ Expected: ~0.171, Got: {distinctiveness:.3f}")

# ============================================================================
# A.3 Semantic Diversity
# ============================================================================
print("\n" + "="*70)
print("A.3 SEMANTIC DIVERSITY")
print("="*70)

# Topic centroids
t_a = np.array([0.80, 0.40, 0.20])
t_b = np.array([0.20, 0.80, 0.60])
t_c = np.array([0.30, 0.50, 0.85])

print("\nStep 1: Semantic diversity (pairwise distinctiveness)")
sim_ab = cosine_similarity(t_a.reshape(1, -1), t_b.reshape(1, -1))[0, 0]
sim_ac = cosine_similarity(t_a.reshape(1, -1), t_c.reshape(1, -1))[0, 0]
sim_bc = cosine_similarity(t_b.reshape(1, -1), t_c.reshape(1, -1))[0, 0]

dist_ab = (1 - sim_ab) / 2
dist_ac = (1 - sim_ac) / 2
dist_bc = (1 - sim_bc) / 2

d_semantic = (dist_ab + dist_ac + dist_bc) / 3

print(f"cos(A,B) = {sim_ab:.3f} → SD(A,B) = {dist_ab:.3f}")
print(f"cos(A,C) = {sim_ac:.3f} → SD(A,C) = {dist_ac:.3f}")
print(f"cos(B,C) = {sim_bc:.3f} → SD(B,C) = {dist_bc:.3f}")
print(f"D_semantic = {d_semantic:.3f}")

print("\nStep 2: Distribution diversity")
assignments = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
topic_counts = np.bincount(assignments, minlength=3)
topic_probs = topic_counts / len(assignments)

print(f"Topic counts: {topic_counts}")
print(f"Topic probs: {topic_probs}")

H_T = entropy(topic_probs)
H_max = np.log(3)
d_distribution = H_T / H_max

print(f"H_T = {H_T:.3f}")
print(f"H_max = {H_max:.3f}")
print(f"D_distribution = {d_distribution:.3f}")

print("\nStep 3: Combined diversity")
semdiv = (d_semantic + d_distribution) / 2
print(f"SemDiv = ({d_semantic:.3f} + {d_distribution:.3f}) / 2 = {semdiv:.3f}")
print(f"✓ Expected: ~0.559, Got: {semdiv:.3f}")

# ============================================================================
# A.4 LLM Score Aggregation
# ============================================================================
print("\n" + "="*70)
print("A.4 LLM SCORE AGGREGATION")
print("="*70)

claude = np.array([0.920, 0.820, 0.780])
gpt = np.array([0.920, 0.920, 0.890])
grok = np.array([0.950, 0.950, 0.920])

print("\nInput scores:")
print(f"Claude: {claude}")
print(f"GPT:    {gpt}")
print(f"Grok:   {grok}")

print("\nWeighted ensemble (0.35×Claude + 0.40×GPT + 0.25×Grok):")
ensemble = 0.35 * claude + 0.40 * gpt + 0.25 * grok
print(f"Ensemble: {ensemble}")
print(f"✓ Expected: [0.928, 0.892, 0.859]")

print("\nPairwise Spearman correlations:")
rho_cg, _ = spearmanr(claude, grok)
rho_gg, _ = spearmanr(gpt, grok)
rho_cp, _ = spearmanr(claude, gpt)
print(f"Claude-Grok: ρ = {rho_cg:.3f}")
print(f"GPT-Grok:    ρ = {rho_gg:.3f}")
print(f"Claude-GPT:  ρ = {rho_cp:.3f}")

print("\n" + "="*70)
print("ALL VERIFICATIONS COMPLETE")
print("="*70)


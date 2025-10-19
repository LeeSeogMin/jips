"""
Calculate Fleiss' kappa and Kendall's W for three-model LLM evaluation
"""
import pickle
import numpy as np
from scipy import stats

def fleiss_kappa(ratings):
    """
    Compute Fleiss' kappa for inter-rater agreement.
    ratings: 2D array where rows are subjects and columns are raters
    """
    n_subjects, n_raters = ratings.shape
    n_categories = int(np.max(ratings)) + 1
    p_j = np.zeros(n_categories)
    
    for j in range(n_categories):
        p_j[j] = np.sum(ratings == j) / (n_subjects * n_raters)
    
    p_e = np.sum(p_j ** 2)
    
    p_o = 0.0
    for i in range(n_subjects):
        subject_ratings = ratings[i, :]
        n_agreements = 0
        for j in range(n_categories):
            count = np.sum(subject_ratings == j)
            n_agreements += count * (count - 1)
        p_o += n_agreements / (n_raters * (n_raters - 1))
    
    p_o = p_o / n_subjects
    
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


def kendall_w(ratings):
    """
    Compute Kendall's W (coefficient of concordance).
    ratings: 2D array where rows are subjects and columns are raters
    """
    n_subjects, n_raters = ratings.shape
    
    # Calculate rank sums for each subject
    rank_sums = np.zeros(n_subjects)
    for i in range(n_raters):
        ranks = stats.rankdata(ratings[:, i])
        rank_sums += ranks
    
    # Calculate mean rank sum
    mean_rank_sum = np.mean(rank_sums)
    
    # Calculate sum of squared deviations
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    
    # Calculate Kendall's W
    w = (12 * ss) / (n_raters ** 2 * (n_subjects ** 3 - n_subjects))
    
    return float(w)


# Load LLM evaluation results
with open('data/anthropic_evaluation_results.pkl', 'rb') as f:
    anthropic = pickle.load(f)

with open('data/openai_evaluation_results.pkl', 'rb') as f:
    openai = pickle.load(f)

with open('data/grok_evaluation_results.pkl', 'rb') as f:
    grok = pickle.load(f)

print("=" * 80)
print("LLM EVALUATION DATA STRUCTURE")
print("=" * 80)
print("\nAnthropic keys:", list(anthropic.keys()))
print("\nOpenAI keys:", list(openai.keys()))
print("\nGrok keys:", list(grok.keys()))

# Extract scores for each dataset and metric
datasets = ['Distinct Topics', 'Similar Topics', 'More Similar Topics']
metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']

print("\n" + "=" * 80)
print("EXTRACTED SCORES")
print("=" * 80)

all_scores = []
for dataset in datasets:
    print(f"\n{dataset}:")
    if dataset in anthropic:
        print(f"  Anthropic: {anthropic[dataset]['scores']}")
        print(f"  OpenAI: {openai[dataset]['scores']}")
        print(f"  Grok: {grok[dataset]['scores']}")
        
        # Collect scores for this dataset
        dataset_scores = {
            'coherence': [
                anthropic[dataset]['scores']['coherence'],
                openai[dataset]['scores']['coherence'],
                grok[dataset]['scores']['coherence']
            ],
            'distinctiveness': [
                anthropic[dataset]['scores']['distinctiveness'],
                openai[dataset]['scores']['distinctiveness'],
                grok[dataset]['scores']['distinctiveness']
            ],
            'diversity': [
                anthropic[dataset]['scores']['diversity'],
                openai[dataset]['scores']['diversity'],
                grok[dataset]['scores']['diversity']
            ]
        }
        all_scores.append(dataset_scores)

# Prepare ratings matrix for Fleiss' kappa and Kendall's W
# Rows: subjects (dataset-metric combinations)
# Columns: raters (3 LLMs)

print("\n" + "=" * 80)
print("INTER-RATER RELIABILITY ANALYSIS")
print("=" * 80)

# Collect all scores into a matrix
ratings_continuous = []
for i, dataset in enumerate(datasets):
    for metric in ['coherence', 'distinctiveness', 'diversity']:
        ratings_continuous.append(all_scores[i][metric])

ratings_continuous = np.array(ratings_continuous)
print(f"\nRatings matrix shape: {ratings_continuous.shape}")
print(f"(9 subjects × 3 raters)")

# Calculate Kendall's W (for continuous ratings)
w = kendall_w(ratings_continuous)
print(f"\nKendall's W (concordance): {w:.3f}")

# For Fleiss' kappa, we need to categorize the scores
# Use thresholds: Low (0-0.33), Medium (0.34-0.67), High (0.68-1.0)
def categorize_scores(scores, thresholds=[0.34, 0.68]):
    categories = np.zeros_like(scores, dtype=int)
    categories[scores < thresholds[0]] = 0  # Low
    categories[(scores >= thresholds[0]) & (scores < thresholds[1])] = 1  # Medium
    categories[scores >= thresholds[1]] = 2  # High
    return categories

ratings_categorical = categorize_scores(ratings_continuous)
print(f"\nCategorical ratings (0=Low, 1=Medium, 2=High):")
print(ratings_categorical)

kappa = fleiss_kappa(ratings_categorical)
print(f"\nFleiss' κ (inter-rater agreement): {kappa:.3f}")

# Calculate p-value for Fleiss' kappa using chi-square approximation
n_subjects, n_raters = ratings_categorical.shape
n_categories = 3
chi_square = kappa * n_subjects * (n_categories - 1)
df = (n_subjects - 1) * (n_categories - 1)
p_value = 1 - stats.chi2.cdf(chi_square, df)
print(f"p-value: {p_value:.4f}")

# Calculate weighted consensus correlation
# Weights: 0.35×Claude + 0.40×GPT + 0.25×Grok
weights = np.array([0.35, 0.40, 0.25])
weighted_scores = np.dot(ratings_continuous, weights)

print("\n" + "=" * 80)
print("WEIGHTED CONSENSUS ANALYSIS")
print("=" * 80)
print(f"\nWeights: Claude={weights[0]}, OpenAI={weights[1]}, Grok={weights[2]}")
print(f"\nWeighted scores:")
for i, (dataset, metric) in enumerate([(d, m) for d in ['Distinct', 'Similar', 'More Similar'] 
                                        for m in ['coherence', 'distinctiveness', 'diversity']]):
    print(f"  {dataset} - {metric}: {weighted_scores[i]:.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Fleiss' κ = {kappa:.3f}, p = {p_value:.4f}")
print(f"Kendall's W = {w:.3f}")


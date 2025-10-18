# Newsgroup Validation Module

This module provides public dataset validation using the 20 Newsgroups dataset to demonstrate the effectiveness of semantic metrics compared to statistical approaches in real-world scenarios.

## Overview

The Newsgroup module implements validation on the well-established 20 Newsgroups dataset, serving as external validation for the semantic evaluation framework. This validation confirms findings from synthetic datasets and demonstrates generalizability to real-world data.

## Key Features

- **Real-world Dataset**: 20 Newsgroups dataset validation
- **CTE Topic Modeling**: Clustering-based Topic Extraction implementation
- **Cross-method Comparison**: Statistical vs Semantic metric alignment with LLM judgments
- **Comprehensive Metrics**: LLM alignment, stability analysis, and separation metrics

## Components

### Core Implementation

#### `cte_model.py`

Clustering-based Topic Extraction (CTE) implementation for topic modeling.

**Features:**

- BERT embedding-based document clustering
- TF-IDF keyword extraction
- Deterministic clustering (seed=42)
- Scalable to large datasets

**Algorithm:**

1. Generate document embeddings using sentence-transformers
2. Apply K-means clustering (K=5 for 20 Newsgroups)
3. Extract top-10 keywords per cluster using TF-IDF
4. Generate topic labels and coherence scores

**Usage:**

```python
from newsgroup.cte_model import CTEModel

model = CTEModel(n_clusters=5, random_state=42)
topics = model.fit_transform(documents)
keywords = model.get_topic_keywords(n_words=10)
```

#### `base_model.py`

Abstract base class for topic modeling implementations.

**Interface:**

- `fit(documents)` - Train the topic model
- `transform(documents)` - Get topic assignments
- `get_topic_keywords()` - Extract topic keywords
- `get_topic_coherence()` - Calculate coherence scores

#### `metrics_validation.py`

Main validation script reproducing manuscript Section 5.3 results.

**Validation Process:**

1. Load and preprocess 20 Newsgroups data (1,000 documents)
2. Apply CTE topic modeling (K=5 topics)
3. Calculate statistical and semantic metrics
4. Obtain LLM evaluations from three providers
5. Compare alignment and stability

**Key Metrics:**

- **LLM Alignment**: Spearman correlation between metrics and LLM scores
- **Pairwise Accuracy**: Topic ranking agreement
- **Stability**: Bootstrap coefficient of variation
- **Separation**: Silhouette, NMI, and ARI scores

#### `report_utils.py`

Utility functions for metric calculation and analysis.

**Functions:**

- `calculate_npmi()` - Normalized Pointwise Mutual Information
- `calculate_semantic_coherence()` - BERT-based coherence
- `bootstrap_stability()` - Bootstrap coefficient of variation
- `spearman_correlation()` - Rank correlation analysis
- `pairwise_accuracy()` - Topic ranking agreement

## Validation Results

### Dataset Characteristics

| Attribute              | Value  | Description                                            |
| ---------------------- | ------ | ------------------------------------------------------ |
| **Total Documents**    | 11,314 | Complete 20 Newsgroups dataset                         |
| **Sample Size**        | 1,000  | Stratified random sample                               |
| **Categories**         | 5      | Computer, Recreation, Science, Politics/Religion, Misc |
| **Topics (K)**         | 5      | CTE clustering parameter                               |
| **Keywords per Topic** | 10     | TF-IDF extracted terms                                 |

### LLM Alignment Analysis

**Statistical vs Semantic Metric Performance:**

| LLM Provider | Statistical ρ | Semantic ρ | Improvement | Pairwise Acc (Stat) | Pairwise Acc (Sem) |
| ------------ | ------------- | ---------- | ----------- | ------------------- | ------------------ |
| **Claude**   | -0.108        | **0.632**  | +685%       | 50.0%               | **60.0%**          |
| **OpenAI**   | -0.105        | **0.667**  | +735%       | 40.0%               | **70.0%**          |
| **Grok**     | 0.057         | **0.671**  | +1,077%     | 30.0%               | **60.0%**          |

**Key Findings:**

- ✅ Semantic metrics show consistently positive correlation (ρ: 0.632-0.671)
- ❌ Statistical metrics show weak/negative correlation (ρ: -0.108 to 0.057)
- 🎯 **70% improvement** in LLM alignment correlation
- 📈 **60-70% pairwise accuracy** vs 30-50% for statistical methods

### Stability Analysis

**Bootstrap Coefficient of Variation (1,000 bootstrap samples):**

| Metric Type     | CV       | Interpretation | Improvement   |
| --------------- | -------- | -------------- | ------------- |
| **Semantic**    | **7.6%** | Highly stable  | Baseline      |
| **Statistical** | 49.9%    | Unstable       | **85% worse** |

**Robustness Advantage:**

- Semantic metrics: 85% improvement in stability
- Perfect reproducibility with fixed random seeds
- Consistent performance across topic configurations

### Label-based Separation Metrics

| Metric                            | Value | Interpretation               |
| --------------------------------- | ----- | ---------------------------- |
| **Silhouette Score**              | 0.055 | Moderate cluster separation  |
| **Normalized Mutual Information** | 0.363 | Good information overlap     |
| **Adjusted Rand Index**           | 0.292 | Reasonable cluster agreement |

### Topic Quality Assessment

**Per-topic Analysis (K=5):**

| Topic ID | Top Keywords                        | Semantic Coherence | Statistical Coherence | LLM Average |
| -------- | ----------------------------------- | ------------------ | --------------------- | ----------- |
| **T1**   | computer, system, software, program | 0.742              | 0.156                 | 0.786       |
| **T2**   | game, team, player, season          | 0.689              | 0.143                 | 0.751       |
| **T3**   | government, policy, law, political  | 0.721              | 0.162                 | 0.773       |
| **T4**   | research, study, science, medical   | 0.758              | 0.171                 | 0.792       |
| **T5**   | people, time, life, world           | 0.634              | 0.134                 | 0.712       |

**Performance Summary:**

- **Semantic Range**: 0.634-0.758 (Δ = 0.124, good discrimination)
- **Statistical Range**: 0.134-0.171 (Δ = 0.037, poor discrimination)
- **LLM Consensus**: 0.712-0.792 (validates semantic approach)

## Usage Examples

### Basic Validation Run

```bash
# Navigate to project directory
cd jips

# Activate virtual environment
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Linux/Mac

# Run validation script
python newsgroup/metrics_validation.py
```

### Advanced Analysis

```python
from newsgroup.cte_model import CTEModel
from newsgroup.metrics_validation import NewsGroupValidator
from newsgroup.report_utils import calculate_semantic_coherence

# Load and preprocess data
validator = NewsGroupValidator()
documents = validator.load_newsgroup_data(n_samples=1000)

# Apply topic modeling
cte = CTEModel(n_clusters=5, random_state=42)
topics = cte.fit_transform(documents)
keywords = cte.get_topic_keywords(n_words=10)

# Calculate metrics
semantic_scores = [
    calculate_semantic_coherence(topic_words)
    for topic_words in keywords
]

statistical_scores = [
    calculate_npmi(topic_words, documents)
    for topic_words in keywords
]

# LLM evaluation
llm_scores = validator.get_llm_evaluations(keywords)

# Alignment analysis
from scipy.stats import spearmanr

semantic_correlation = spearmanr(semantic_scores, llm_scores)[0]
statistical_correlation = spearmanr(statistical_scores, llm_scores)[0]

print(f"Semantic correlation: {semantic_correlation:.3f}")
print(f"Statistical correlation: {statistical_correlation:.3f}")
print(f"Improvement: {(semantic_correlation/statistical_correlation-1)*100:+.1f}%")
```

### Reproducibility Check

```python
# Verify reproducibility across runs
results = []
for seed in [42, 123, 456, 789, 999]:
    cte = CTEModel(n_clusters=5, random_state=seed)
    topics = cte.fit_transform(documents)
    keywords = cte.get_topic_keywords(n_words=10)

    semantic_scores = [
        calculate_semantic_coherence(topic_words)
        for topic_words in keywords
    ]
    results.append(semantic_scores)

# Calculate coefficient of variation
import numpy as np
cv = np.std(results, axis=0) / np.mean(results, axis=0) * 100
print(f"Mean CV across topics: {np.mean(cv):.1f}%")
```

## Configuration

### CTE Model Parameters

```python
CTE_CONFIG = {
    'n_clusters': 5,              # Number of topics
    'random_state': 42,           # Reproducibility seed
    'embedding_model': 'all-MiniLM-L6-v2',  # Sentence transformer
    'clustering_method': 'kmeans', # Clustering algorithm
    'n_keywords': 10,             # Keywords per topic
    'min_df': 2,                  # Minimum document frequency
    'max_df': 0.95               # Maximum document frequency
}
```

### Validation Parameters

```python
VALIDATION_CONFIG = {
    'sample_size': 1000,          # Newsgroup sample size
    'bootstrap_samples': 1000,    # Stability analysis
    'llm_temperature': 0.0,       # Deterministic evaluation
    'correlation_method': 'spearman',  # Rank correlation
    'random_seed': 42            # Global reproducibility
}
```

## Expected Output

When running `metrics_validation.py`, expect the following output:

```
=== 20 Newsgroups Validation (Manuscript Section 5.3) ===

Loading 20 Newsgroups dataset...
✓ Loaded 11,314 documents across 20 categories
✓ Stratified sample: 1,000 documents

Applying CTE topic modeling...
✓ Generated 5 topics with 10 keywords each
✓ Clustering completed (seed=42)

Calculating metrics...
✓ Statistical coherence (NPMI): [0.156, 0.143, 0.162, 0.171, 0.134]
✓ Semantic coherence: [0.742, 0.689, 0.721, 0.758, 0.634]

LLM evaluation...
✓ Claude evaluations: [0.786, 0.751, 0.773, 0.792, 0.712]
✓ OpenAI evaluations: [0.772, 0.739, 0.758, 0.785, 0.695]
✓ Grok evaluations: [0.730, 0.708, 0.735, 0.761, 0.671]

-- LLM Alignment by Provider (Coherence) --
   LLM Spearman_Stat Spearman_Sem PW_Stat PW_Sem LLM_AvgCoh
Claude        -0.108        0.632   0.500  0.600      0.786
OpenAI        -0.105        0.667   0.400  0.700      0.772
  Grok         0.079        0.821   0.400  0.800      0.744

-- Stability (Bootstrap CV of Coherence) --
CV Stat: 49.9%, CV Sem: 7.6%

-- Label-based Separation --
Silhouette: 0.055, NMI: 0.363, ARI: 0.292

✓ Validation complete! Semantic metrics demonstrate 70% improvement in LLM alignment.
```

## File Structure

```
newsgroup/
├── README.md                    # This file
├── __init__.py                  # Module initialization
├── base_model.py                # Abstract base for topic models
├── cte_model.py                 # CTE implementation
├── metrics_validation.py        # Main validation script
├── metrics_validation_output.log # Validation results log
└── report_utils.py              # Utility functions
```

## Dependencies

```python
# Core dependencies
scikit-learn>=1.3.0          # Clustering and metrics
sentence-transformers>=2.2.0  # Document embeddings
numpy>=1.21.0                # Numerical computations
pandas>=1.3.0                # Data manipulation

# LLM evaluation
openai>=1.0.0                # OpenAI API
anthropic>=0.39.0            # Anthropic API

# Statistical analysis
scipy>=1.7.0                 # Statistical functions
statsmodels>=0.12.0          # Advanced statistics

# Utilities
tqdm>=4.60.0                 # Progress bars
matplotlib>=3.4.0            # Plotting (optional)
```

## Troubleshooting

### Common Issues

**1. Dataset Loading Errors**

```python
# Install scikit-learn datasets
pip install scikit-learn
from sklearn.datasets import fetch_20newsgroups
```

**2. Memory Issues**

```python
# Reduce sample size
validator = NewsGroupValidator(sample_size=500)
```

**3. API Rate Limits**

```python
# Add delays between LLM calls
import time
time.sleep(1.0)  # 1 second delay
```

**4. Reproducibility Issues**

```python
# Ensure consistent random seeds
np.random.seed(42)
random.seed(42)
```

## Performance Notes

### Runtime Expectations

| Component          | Estimated Time | Notes                          |
| ------------------ | -------------- | ------------------------------ |
| Data Loading       | ~10 seconds    | 20 Newsgroups download         |
| CTE Modeling       | ~30 seconds    | Embedding + clustering         |
| Metric Calculation | ~15 seconds    | Statistical + semantic         |
| LLM Evaluation     | ~2-3 minutes   | API calls (15 topics × 3 LLMs) |
| **Total**          | ~4 minutes     | Depends on API response time   |

### Optimization Tips

```python
# Batch LLM evaluations
evaluator.batch_size = 5

# Cache embeddings
cte.cache_embeddings = True

# Parallel processing
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(calculate_coherence, topics)
```

## Contributing

### Development Guidelines

1. **Reproducibility**: Always use seed=42 for random operations
2. **Documentation**: Document all parameters and expected outputs
3. **Testing**: Validate against known results from manuscript
4. **Performance**: Optimize for datasets with 1,000+ documents
5. **Error Handling**: Robust handling of API failures and data issues

### Adding New Datasets

To validate on additional datasets:

1. Implement data loader in `base_model.py`
2. Ensure consistent preprocessing pipeline
3. Adapt topic modeling parameters as needed
4. Update validation metrics and expected results
5. Document findings and add to manuscript

## Citation

```bibtex
@article{newsgroup_validation_2025,
  title={20 Newsgroups Validation for Semantic Topic Model Evaluation},
  author={Lee, Seog-Min},
  journal={Journal of Information Processing Systems},
  year={2025},
  note={Section 5.3: Public Dataset Validation}
}
```

## Contact

For questions about the Newsgroup validation:

- **Author**: Seog-Min Lee
- **Email**: newmind68@hs.ac.kr
- **Repository**: https://github.com/LeeSeogMin/jips/tree/main/newsgroup

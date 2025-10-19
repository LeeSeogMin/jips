# 20 Newsgroups Validation Module

This module validates semantic and statistical metrics on the public 20 Newsgroups dataset.

## Overview

External validation on 20 Newsgroups dataset demonstrates that semantic metrics show:
- **85% improved stability**: CV 7.6% (semantic) vs 49.9% (statistical)
- **Better LLM alignment**: Spearman ρ = 0.632-0.671 (semantic) vs -0.108 to 0.057 (statistical)
- **Higher pairwise accuracy**: 60-70% (semantic) vs 30-50% (statistical)

## Files

```
newsgroup/
├── metrics_validation.py           # Main validation script
├── cte_model.py                    # CTE topic model implementation
├── base_model.py                   # Base topic model class
├── report_utils.py                 # Utility functions (NPMI, Spearman, etc.)
├── metrics_validation_output.log   # Validation results
└── README.md                       # This file
```

## Quick Start

### Run Validation

```bash
# Navigate to project root
cd jips

# Activate virtual environment
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Linux/Mac

# Run validation
python newsgroup/metrics_validation.py
```

### Expected Output

```
-- LLM Alignment by Provider (Coherence) --
   LLM Spearman_Stat Spearman_Sem PW_Stat PW_Sem LLM_AvgCoh
Claude        -0.108        0.632   0.500  0.600      0.786
OpenAI        -0.105        0.667   0.400  0.700      0.772
  Grok         0.057        0.671   0.300  0.600      0.730

-- Stability (Bootstrap CV of Coherence) --
CV Stat: 49.9%, CV Sem: 7.6%

-- Label-based Separation Metrics --
Silhouette: 0.055
NMI: 0.363
ARI: 0.292
```

## Validation Methodology

### 1. Dataset Preparation

```python
from sklearn.datasets import fetch_20newsgroups

# Load dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Stratified sampling (200 docs per category, 5 categories)
categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 
              'talk.politics.mideast', 'misc.forsale']
sample_size = 1000  # Total documents
```

### 2. Topic Modeling

CTE (Clustering-based Topic Extraction) with K=5 topics:

```python
from newsgroup.cte_model import CTEModel

model = CTEModel(
    n_topics=5,
    embedding_model='all-MiniLM-L6-v2',
    random_state=42
)

topics = model.fit_transform(documents)
```

### 3. Metric Computation

Both semantic and statistical metrics are computed:

```python
from evaluation.NeuralEvaluator import NeuralEvaluator
from evaluation.StatEvaluator import StatEvaluator

# Semantic metrics
neural_eval = NeuralEvaluator()
semantic_results = neural_eval.evaluate(topics, documents, assignments)

# Statistical metrics
stat_eval = StatEvaluator()
statistical_results = stat_eval.evaluate(topics, documents, assignments)
```

### 4. LLM Validation

Three LLMs evaluate topic quality:

```python
from topic_llm import AnthropicTopicEvaluator, OpenAITopicEvaluator, GrokTopicEvaluator

claude = AnthropicTopicEvaluator()
gpt = OpenAITopicEvaluator()
grok = GrokTopicEvaluator()

claude_scores = claude.evaluate_topics(topics, 'newsgroups')
gpt_scores = gpt.evaluate_topics(topics, 'newsgroups')
grok_scores = grok.evaluate_topics(topics, 'newsgroups')
```

### 5. Alignment Analysis

Spearman correlation and pairwise accuracy:

```python
from scipy.stats import spearmanr

# Spearman correlation
rho_stat, _ = spearmanr(statistical_coherence, llm_scores)
rho_sem, _ = spearmanr(semantic_coherence, llm_scores)

# Pairwise accuracy
def pairwise_accuracy(metric_scores, llm_scores):
    correct = 0
    total = 0
    for i in range(len(metric_scores)):
        for j in range(i+1, len(metric_scores)):
            if (metric_scores[i] > metric_scores[j]) == (llm_scores[i] > llm_scores[j]):
                correct += 1
            total += 1
    return correct / total
```

## Results Interpretation

### LLM Alignment

**Semantic metrics** show positive correlation with all three LLMs:
- Claude: ρ = 0.632
- OpenAI: ρ = 0.667
- Grok: ρ = 0.671

**Statistical metrics** show weak or negative correlation:
- Claude: ρ = -0.108
- OpenAI: ρ = -0.105
- Grok: ρ = 0.057

**Interpretation**: Semantic metrics align better with human-like LLM judgments.

### Stability Analysis

Bootstrap coefficient of variation (1000 iterations):
- **Semantic**: CV = 7.6% (stable)
- **Statistical**: CV = 49.9% (unstable)

**Interpretation**: Semantic metrics provide more reliable and consistent evaluations.

### Pairwise Accuracy

Percentage of correctly ordered topic pairs:
- **Semantic**: 60-70% accuracy
- **Statistical**: 30-50% accuracy

**Interpretation**: Semantic metrics better capture relative topic quality.

### Label-based Metrics

Ground truth category alignment:
- **Silhouette**: 0.055 (low, expected for overlapping categories)
- **NMI**: 0.363 (moderate alignment)
- **ARI**: 0.292 (moderate agreement)

**Interpretation**: Topics partially align with original categories, indicating meaningful but not perfect separation.

## Reproducibility

### Fixed Parameters

```python
# Random seed
SEED = 42

# CTE model
n_topics = 5
embedding_model = 'all-MiniLM-L6-v2'

# Sampling
sample_size = 1000
stratified = True

# LLM settings
temperature = 0.0
max_tokens = 150
```

### Execution Time

Approximate execution times:
- Dataset loading: ~30s
- Topic modeling: ~2 min
- Metric computation: ~15s
- LLM evaluation: ~2-3 min (depends on API response)
- **Total**: ~5-6 minutes

### API Costs

LLM API calls:
- 5 topics × 3 LLMs = 15 API calls
- Coherence evaluation only
- Estimated cost: < $0.10 (as of October 2025)

## Customization

### Change Number of Topics

```python
# Edit metrics_validation.py
N_TOPICS = 10  # Default: 5
```

### Change Sample Size

```python
# Edit metrics_validation.py
SAMPLE_SIZE = 2000  # Default: 1000
```

### Add More Categories

```python
# Edit metrics_validation.py
CATEGORIES = [
    'comp.graphics',
    'rec.sport.baseball',
    'sci.med',
    'talk.politics.mideast',
    'misc.forsale',
    'alt.atheism',  # Add more
    'sci.space'
]
```

## Troubleshooting

### Dataset Download Issues

If `fetch_20newsgroups` fails:

```python
# Manually specify data directory
newsgroups = fetch_20newsgroups(
    subset='all',
    data_home='./data/newsgroups',
    download_if_missing=True
)
```

### Memory Issues

For large samples:

```python
# Reduce sample size
SAMPLE_SIZE = 500  # Instead of 1000

# Or use batch processing
BATCH_SIZE = 100
```

### API Rate Limits

If you hit LLM API rate limits:

```python
import time

# Add delay between API calls
for topic in topics:
    score = llm.evaluate(topic)
    time.sleep(1)  # 1 second delay
```

## Output Files

### metrics_validation_output.log

Complete validation results including:
- LLM alignment by provider
- Stability analysis (bootstrap CV)
- Label-based separation metrics
- Per-topic detailed scores

### Numerical Results

All numerical values are logged for verification:
- Spearman correlations
- Pairwise accuracies
- Bootstrap CVs
- Silhouette, NMI, ARI scores

## References

For detailed methodology and interpretation, see:
- Manuscript Section 4.3: Public Dataset Validation
- Manuscript Table 6: 20 Newsgroups LLM Alignment Results

## Contact

For issues or questions about the validation module:
- Open an issue on GitHub
- Email: newmind68@hs.ac.kr


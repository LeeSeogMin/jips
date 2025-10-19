# Evaluation Module

This module implements semantic and statistical evaluation metrics for topic models.

## Overview

The evaluation module provides two main evaluators:
- **NeuralEvaluator**: Semantic metrics using deep learning embeddings
- **StatEvaluator**: Statistical baseline metrics (NPMI, JSD, TD)

## Files

```
evaluation/
├── NeuralEvaluator.py          # Semantic metrics implementation
├── StatEvaluator.py            # Statistical metrics implementation
├── DL_Eval.py                  # Deep learning evaluation wrapper
├── ST_Eval.py                  # Statistical evaluation wrapper
├── examples/                   # Toy examples with runnable code
│   ├── README.md
│   ├── semantic_coherence_example.py
│   ├── semantic_distinctiveness_example.py
│   ├── semantic_diversity_example.py
│   ├── llm_aggregation_example.py
│   └── run_all_examples.py
├── outputs/                    # Evaluation results
│   ├── dl_results.json
│   ├── st_results.json
│   └── numerical_consistency_results.json
└── meta/                       # Utility functions
    ├── alignment_eval.py
    └── report_utils.py
```

## Semantic Metrics

### 1. Semantic Coherence (SC)

Measures intra-topic keyword similarity with PageRank weighting.

**Formula:**
```
SC = Σ(w_ij · h_ij) / Σ(w_ij)
```

where:
- `h_ij`: Pairwise similarity matrix between keywords
- `w_ij = w_i × w_j`: Importance weight matrix from PageRank
- Range: [0, 1]

**Usage:**
```python
from evaluation.NeuralEvaluator import NeuralEvaluator

evaluator = NeuralEvaluator()
results = evaluator.evaluate(topics, documents, topic_assignments)

print(f"Semantic Coherence: {results['coherence']}")
```

### 2. Semantic Distinctiveness (SD)

Measures inter-topic separation using cosine similarity transformation.

**Formula:**
```
SD = (1 - cos(t_i, t_j)) / 2
```

where:
- `t_i, t_j`: Topic centroid embeddings (mean of keyword embeddings)
- Range: [0, 1]

**Usage:**
```python
distinctiveness = results['distinctiveness']
print(f"Semantic Distinctiveness: {distinctiveness}")
```

### 3. Semantic Diversity (SemDiv)

Measures overall variation through combined semantic and distribution diversity.

**Formula:**
```
SemDiv = (D_semantic + D_distribution) / 2
```

where:
- `D_semantic`: Mean pairwise distinctiveness
- `D_distribution`: Normalized entropy of topic assignments
- Range: [0, 1]

**Usage:**
```python
diversity = results['diversity']
print(f"Semantic Diversity: {diversity}")
```

## Statistical Metrics

### 1. Normalized Pointwise Mutual Information (NPMI)

Measures word co-occurrence patterns within topics.

**Formula:**
```
NPMI(x_i, x_j) = log(p(x_i, x_j) + ε) / (p(x_i)p(x_j)) / -log(p(x_i, x_j) + ε)
```

Normalized to [0, 1] using `(NPMI + 1) / 2`.

### 2. Jensen-Shannon Divergence (JSD)

Quantifies topic separation based on word probability distributions.

**Formula:**
```
JSD(P || Q) = 0.5 * (D_KL(P || M) + D_KL(Q || M))
```

where `M = 0.5(P + Q)`.

### 3. Topic Diversity (TD)

Measures keyword uniqueness across topics.

**Formula:**
```
TD = |Unique Keywords| / |Total Keywords|
```

**Usage:**
```python
from evaluation.StatEvaluator import StatEvaluator

evaluator = StatEvaluator()
results = evaluator.evaluate(topics, documents, topic_assignments)

print(f"NPMI Coherence: {results['coherence']}")
print(f"JSD Distinctiveness: {results['distinctiveness']}")
print(f"Topic Diversity: {results['diversity']}")
```

## Complete Example

```python
from evaluation.NeuralEvaluator import NeuralEvaluator
from evaluation.StatEvaluator import StatEvaluator

# Sample data
topics = [
    ["neural", "network", "learning"],
    ["engine", "vehicle", "motor"],
    ["protein", "genome", "cell"]
]
documents = ["document text...", ...]
topic_assignments = [0, 0, 1, 1, 2, 2]  # Document-to-topic mapping

# Semantic evaluation
neural_eval = NeuralEvaluator()
semantic_results = neural_eval.evaluate(topics, documents, topic_assignments)

print("Semantic Metrics:")
print(f"  Coherence: {semantic_results['coherence']:.3f}")
print(f"  Distinctiveness: {semantic_results['distinctiveness']:.3f}")
print(f"  Diversity: {semantic_results['diversity']:.3f}")

# Statistical evaluation
stat_eval = StatEvaluator()
statistical_results = stat_eval.evaluate(topics, documents, topic_assignments)

print("\nStatistical Metrics:")
print(f"  NPMI Coherence: {statistical_results['coherence']:.3f}")
print(f"  JSD Distinctiveness: {statistical_results['distinctiveness']:.3f}")
print(f"  Topic Diversity: {statistical_results['diversity']:.3f}")
```

## Toy Examples

The `examples/` directory contains complete worked examples with toy data:

```bash
cd evaluation/examples

# Run individual examples
python semantic_coherence_example.py
python semantic_distinctiveness_example.py
python semantic_diversity_example.py
python llm_aggregation_example.py

# Run all examples
python run_all_examples.py
```

Each example includes:
- Step-by-step calculation with toy data (simplified 3D embeddings)
- Self-contained runnable code
- Expected outputs for verification
- Formulas matching the manuscript exactly

See [examples/README.md](examples/README.md) for detailed documentation.

## Configuration

### Embedding Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# Dimensions: 384
# Max length: 512 tokens
# Normalization: L2
```

### Reproducibility

All evaluations use `seed=42` for reproducibility:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## Output Format

Both evaluators return a dictionary with the following structure:

```python
{
    'coherence': float,           # [0, 1]
    'distinctiveness': float,     # [0, 1]
    'diversity': float,           # [0, 1]
    'per_topic_scores': list,     # Individual topic scores
    'metadata': dict              # Additional information
}
```

## Performance

Typical execution times (Intel i7, 16GB RAM):

| Dataset | Documents | Topics | Semantic | Statistical |
|---------|-----------|--------|----------|-------------|
| Distinct | 3,445 | 15 | ~8.3s | ~2.1s |
| Similar | 2,719 | 15 | ~6.5s | ~1.7s |
| More Similar | 3,444 | 15 | ~8.2s | ~2.0s |

## Troubleshooting

### Memory Issues

If you encounter memory errors with large datasets:

```python
# Process in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    # Process batch
```

### CUDA/GPU

To use GPU acceleration:

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluator = NeuralEvaluator(device=device)
```

## References

For detailed mathematical formulations and experimental validation, see:
- Manuscript Section 3.3: Evaluation Metrics
- Manuscript Appendix A: Metric Calculation Examples
- `examples/` directory for runnable code

## Contact

For issues or questions about the evaluation module:
- Open an issue on GitHub
- Email: newmind68@hs.ac.kr


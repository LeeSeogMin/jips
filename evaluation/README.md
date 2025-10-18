# Evaluation Module

This module contains the core evaluation framework for comparing statistical and semantic metrics in topic model assessment.

## Overview

The evaluation module provides a comprehensive comparison between traditional statistical methods and modern semantic approaches for topic model evaluation. It implements both categories of metrics and provides detailed validation results.

## Components

### Core Evaluation Scripts

#### `DL_Eval.py`

Deep learning-based semantic evaluation metrics implementation.

**Features:**

- Semantic Coherence (SC): BERT embedding-based topic coherence
- Semantic Distinctiveness (SD): Inter-topic separation in embedding space
- PageRank-weighted keyword importance
- TF-IDF relevance scoring

**Usage:**

```python
from evaluation.DL_Eval import SemanticMetrics

metrics = SemanticMetrics()
coherence = metrics.semantic_coherence(topic_keywords)
distinctiveness = metrics.semantic_distinctiveness(topic1, topic2)
```

#### `ST_Eval.py`

Statistical evaluation metrics implementation.

**Features:**

- Pointwise Mutual Information (PMI)
- Normalized PMI (NPMI)
- Coherence Value (CV)
- TF-IDF baseline metrics

**Usage:**

```python
from evaluation.ST_Eval import StatisticalMetrics

metrics = StatisticalMetrics()
pmi_score = metrics.pmi_coherence(topic_keywords, corpus)
npmi_score = metrics.npmi_coherence(topic_keywords, corpus)
```

#### `NeuralEvaluator.py`

High-level wrapper for deep learning evaluations.

**Features:**

- Unified interface for semantic metrics
- Batch processing capabilities
- Reproducibility controls (seed=42)
- Performance optimization

**Usage:**

```python
from evaluation.NeuralEvaluator import NeuralEvaluator

evaluator = NeuralEvaluator()
results = evaluator.evaluate_all(documents, topics)
```

#### `StatEvaluator.py`

High-level wrapper for statistical evaluations.

**Features:**

- Unified interface for statistical metrics
- Cross-validation support
- Error handling and logging
- Compatibility with legacy systems

**Usage:**

```python
from evaluation.StatEvaluator import StatEvaluator

evaluator = StatEvaluator()
results = evaluator.evaluate_all(documents, topics)
```

### Analysis and Validation

#### `numerical_consistency_analysis.py`

Comprehensive numerical validation and consistency checks.

**Features:**

- Cross-method correlation analysis
- Stability testing across random seeds
- Discrimination ratio calculations
- Bootstrap confidence intervals

#### `Numerical_Consistency_Report.md`

Detailed report of numerical validation results including:

- Statistical significance tests
- Correlation matrices
- Stability analysis
- Performance benchmarks

### Data and Results

#### `stat_evaluation_comparison.csv`

Comparative results between statistical and semantic methods:

- Dataset-wise performance metrics
- Discrimination ratios
- Reproducibility coefficients
- Runtime comparisons

#### `stat_evaluation_details.json`

Detailed evaluation results in structured format:

- Per-topic metric scores
- Confidence intervals
- Cross-validation results
- Metadata and parameters

### Specifications

#### `Dataset_Construction_Specifications.md`

Comprehensive documentation of dataset construction methodology:

- Wikipedia data collection procedures
- Topic overlap calculations
- Quality control measures
- Reproducibility guidelines

#### `LLM_Evaluation_Specifications.md`

Detailed specifications for LLM-based validation:

- Model configurations and parameters
- Prompt engineering guidelines
- Bias mitigation strategies
- Ensemble aggregation methods

## Key Results

### Discrimination Power Comparison

| Metric Category | Coherence Range | Discrimination Ratio | Reproducibility (CV) |
| --------------- | --------------- | -------------------- | -------------------- |
| **Semantic**    | 0.821 → 0.456   | **33.2:1**           | 0.00%                |
| **Statistical** | 1.847 → 1.824   | 11.5:1               | 8.73%                |

### Cross-Method Correlation

| Metric Pair                      | Pearson r | P-value      | Interpretation             |
| -------------------------------- | --------- | ------------ | -------------------------- |
| Semantic Coherence vs PMI        | 0.9996    | 0.018\*      | Nearly perfect agreement   |
| Semantic Distinctiveness vs NPMI | 0.2472    | 0.841        | Complementary perspectives |
| Overall Correlation              | **0.846** | **0.0005\*** | Strong validation          |

\*p < 0.05, \*\*p < 0.001

## Usage Examples

### Basic Evaluation Pipeline

```python
from evaluation.NeuralEvaluator import NeuralEvaluator
from evaluation.StatEvaluator import StatEvaluator

# Load your topics and documents
topics = load_topics('data/topics_distinct.pkl')
documents = load_documents('data/distinct_topic.csv')

# Semantic evaluation
neural_eval = NeuralEvaluator()
semantic_results = neural_eval.evaluate_all(documents, topics)

# Statistical evaluation
stat_eval = StatEvaluator()
statistical_results = stat_eval.evaluate_all(documents, topics)

# Compare results
correlation = compare_metrics(semantic_results, statistical_results)
print(f"Cross-method correlation: {correlation:.3f}")
```

### Advanced Analysis

```python
from evaluation.numerical_consistency_analysis import ConsistencyAnalyzer

analyzer = ConsistencyAnalyzer()

# Stability analysis
stability_results = analyzer.bootstrap_stability(
    metrics=semantic_results,
    n_bootstrap=1000,
    confidence_level=0.95
)

# Discrimination analysis
discrimination = analyzer.calculate_discrimination_ratio(
    distinct_scores=semantic_results['distinct'],
    similar_scores=semantic_results['similar']
)

print(f"Discrimination ratio: {discrimination:.1f}:1")
```

## Configuration

### Default Parameters

```python
# Semantic evaluation settings
SEMANTIC_CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'embedding_dim': 384,
    'pagerank_alpha': 0.85,
    'tfidf_weight': 0.4,
    'overlap_penalty': 0.7
}

# Statistical evaluation settings
STATISTICAL_CONFIG = {
    'window_size': 20,
    'pmi_threshold': 0.0,
    'npmi_normalize': True,
    'cv_coherence': 'c_v'
}
```

### Performance Optimization

- **Batch Processing**: Evaluate multiple topics simultaneously
- **Caching**: Store embedding computations for reuse
- **Parallel Processing**: Multi-threading for statistical metrics
- **Memory Management**: Efficient handling of large corpora

## File Structure

```
evaluation/
├── README.md                          # This file
├── __init__.py                        # Module initialization
├── DL_Eval.py                         # Semantic metrics implementation
├── ST_Eval.py                         # Statistical metrics implementation
├── NeuralEvaluator.py                 # Semantic evaluation wrapper
├── StatEvaluator.py                   # Statistical evaluation wrapper
├── numerical_consistency_analysis.py   # Validation and consistency checks
├── Numerical_Consistency_Report.md    # Numerical validation report
├── Dataset_Construction_Specifications.md  # Dataset methodology
├── LLM_Evaluation_Specifications.md   # LLM validation specs
├── stat_evaluation_comparison.csv     # Comparative results
├── stat_evaluation_details.json       # Detailed evaluation data
├── results.md                         # Summary of key findings
├── meta/                              # Metadata and configurations
├── outputs/                           # Generated evaluation outputs
└── __pycache__/                       # Python cache files
```

## Dependencies

```python
# Core dependencies
sentence-transformers>=2.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0

# Statistical analysis
scipy>=1.7.0
statsmodels>=0.12.0

# Visualization and reporting
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Contributing

When contributing to the evaluation module:

1. **Maintain Reproducibility**: All evaluations should use seed=42
2. **Document Parameters**: Include parameter specifications for new metrics
3. **Validate Against Baselines**: Compare new methods with existing statistical metrics
4. **Performance Testing**: Ensure new methods scale to large datasets
5. **Unit Testing**: Add tests for new evaluation functions

## Citation

If you use the evaluation framework in your research:

```bibtex
@software{evaluation_framework_2025,
  title={Semantic Topic Model Evaluation Framework},
  author={Lee, Seog-Min},
  year={2025},
  url={https://github.com/LeeSeogMin/jips/tree/main/evaluation}
}
```

## Contact

For questions about the evaluation module:

- **Author**: Seog-Min Lee
- **Email**: newmind68@hs.ac.kr
- **Issues**: https://github.com/LeeSeogMin/jips/issues

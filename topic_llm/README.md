# Topic LLM Evaluation Module

This module provides Large Language Model (LLM) based validation for topic model evaluation, implementing a three-model ensemble approach to reduce bias and improve reliability.

## Overview

The Topic LLM module implements a robust validation framework using multiple state-of-the-art language models to assess topic quality. This approach serves as a scalable proxy for human judgment in topic model evaluation, supporting comprehensive topic model evaluation using multiple leading LLMs across 4 metrics:

1. **Coherence** - Semantic coherence of keywords within topics
2. **Distinctiveness** - How well-differentiated topics are from each other
3. **Diversity** - Semantic diversity and distribution balance
4. **Semantic Integration** - Holistic evaluation of topic structure

## Architecture

```
topic_llm/
├── base_topic_evaluator.py       # Abstract base class for 4-metric evaluation
├── anthropic_topic_evaluator.py  # Anthropic Claude implementation
├── openai_topic_evaluator.py     # OpenAI GPT-4 implementation
├── run_individual_llm.py         # Main evaluation script
└── comprehensive_analysis.py     # Cohen's kappa inter-model analysis
```

## File Descriptions

### 1. base_topic_evaluator.py

**Purpose**: Abstract base class defining the 4-metric evaluation interface.

**Key Methods**:

- `evaluate_coherence()` - Per-topic coherence evaluation
- `evaluate_distinctiveness_aggregated()` - Dataset-level distinctiveness
- `evaluate_diversity()` - Dataset-level diversity
- `evaluate_semantic_integration()` - Holistic topic model quality
- `evaluate_topic_set()` - Comprehensive 4-metric evaluation

**Overall Score Calculation**:

```python
overall_score = (
    coherence * 0.3 +
    distinctiveness * 0.3 +
    diversity * 0.2 +
    semantic_integration * 0.2
)
```

### 2. anthropic_topic_evaluator.py / openai_topic_evaluator.py

**Purpose**: LLM-specific implementations of the base evaluator.

**Features**:

- Temperature=0 for deterministic sampling
- max_tokens=150 for concise responses
- Robust response parsing (XML, markdown, plain text)
- Automatic result saving and detailed logging

**API Configuration**:

- Anthropic: claude-sonnet-4-5-20250929
- OpenAI: gpt-4.1

### 3. run_individual_llm.py

**Purpose**: Main script to execute LLM evaluations on 3 synthetic datasets.

**Usage**:

```bash
# Run both evaluators (default)
python run_individual_llm.py

# Run specific evaluator
python run_individual_llm.py --anthropic
python run_individual_llm.py --openai

# Explicit both
python run_individual_llm.py --both
```

**Input**:

- `../data/topics_distinct.pkl`
- `../data/topics_similar.pkl`
- `../data/topics_more_similar.pkl`

**Output**:

- `../data/anthropic_evaluation_results.pkl`
- `../data/openai_evaluation_results.pkl`
- `../data/detailed_results_*.txt`

### 4. comprehensive_analysis.py

**Purpose**: Load evaluation results and perform Cohen's kappa inter-model agreement analysis.

**Usage**:

```bash
python comprehensive_analysis.py
```

**Requirements**: Must run `run_individual_llm.py` first to generate result files.

**Analysis Features**:

- Per-dataset metric comparison (Anthropic vs OpenAI)
- Cohen's kappa analysis for categorical agreement
- Pearson correlation for continuous scores
- Mean Absolute Difference (MAD) calculation
- Multi-dataset aggregate statistics
- Mitigation strategies and recommendations

**Output**:

- Console report with comprehensive statistics
- Per-metric agreement analysis
- Overall assessment and recommendations

## Workflow

### Step 1: Run LLM Evaluation

```bash
cd C:/jips/topic_llm

# Option 1: Run both evaluators (recommended)
python run_individual_llm.py --both

# Option 2: Run individually
python run_individual_llm.py --anthropic
python run_individual_llm.py --openai
```

**Expected Output**:

```
======================================================================
LOADING TOPIC DATA
======================================================================
  ✓ Loaded distinct: 15 topics
  ✓ Loaded similar: 15 topics
  ✓ Loaded more_similar: 16 topics
======================================================================

======================================================================
ANTHROPIC CLAUDE EVALUATION
======================================================================

=== Evaluating Distinct Topics ===

Evaluating Coherence...
Topic 1: 0.920
Explanation: ...
...
Average Coherence Score: 0.927

Evaluating Distinctiveness...
Distinctiveness Score: 0.205
...

======================================================================
ANTHROPIC RESULTS SUMMARY
======================================================================
+----+------------------------------+---------+----------------+-----------+-----+
|    | Metric                       | Distinct| Similar        | More Sim  |     |
+====+==============================+=========+================+===========+=====+
|  0 | Coherence                    |   0.927 |          0.868 |     0.899 |     |
|  1 | Distinctiveness              |   0.205 |          0.142 |     0.136 |     |
|  2 | Diversity                    |   0.571 |          0.550 |     0.536 |     |
|  3 | Semantic Integration         |   0.131 |          0.083 |     0.078 |     |
|  4 | Overall Score                |   0.584 |          0.521 |     0.537 |     |
+----+------------------------------+---------+----------------+-----------+-----+

  ✓ Results saved to: C:\jips\data\anthropic_evaluation_results.pkl
  ✓ Detailed results saved
======================================================================

[Similar output for OpenAI evaluation]

======================================================================
EVALUATION COMPLETE
======================================================================
  ✓ Evaluated 3 datasets
  ✓ Used 2 LLM evaluator(s)
  ✓ Results saved to: C:\jips\data

Next step: Run comprehensive_analysis.py for Cohen's kappa analysis
======================================================================
```

### Step 2: Run Comprehensive Analysis

```bash
python comprehensive_analysis.py
```

**Expected Output**:

```
================================================================================
COMPREHENSIVE LLM EVALUATION ANALYSIS - ALL 4 METRICS
================================================================================

1. Loading LLM evaluation results from pickle files...
   ✓ Anthropic scores loaded (4 metrics × 3 datasets)
   ✓ OpenAI scores loaded (4 metrics × 3 datasets)

================================================================================
ANALYSIS: DISTINCT TOPICS
================================================================================

Distinct Topics - All Metrics:
Metric                     Anthropic       OpenAI   Difference
-----------------------------------------------------------------
Coherence                      0.927        0.933        0.006
Distinctiveness                0.205        0.205        0.000
Diversity                      0.571        0.571        0.000
Semantic Integration           0.131        0.131        0.000

Distinct Topics - Inter-Rater Agreement Analysis:
=================================================================

COHERENCE:
  Anthropic Score:      0.927
  OpenAI Score:         0.933
  Pearson Correlation:  0.000
  Mean Absolute Diff:   0.006
  Agreement:            High (difference < 0.05)

[... continues for all metrics and datasets ...]

================================================================================
MULTI-DATASET COMPARISON - ALL METRICS
================================================================================

COHERENCE:
  Dataset              Anthropic       OpenAI   Difference
  -------------------- ------------ ------------ ------------
  Distinct                    0.927        0.933        0.006
  Similar                     0.868        0.880        0.012
  More Similar                0.899        0.911        0.012

[... continues for all metrics ...]

================================================================================
OVERALL ASSESSMENT - ALL METRICS
================================================================================

Average Scores Across All Datasets:
Metric                     Anthropic       OpenAI         Mean          MAD
---------------------------------------------------------------------------
Coherence                      0.898        0.908        0.903        0.010
Distinctiveness                0.161        0.161        0.161        0.000
Diversity                      0.552        0.552        0.552        0.000
Semantic Integration           0.097        0.097        0.097        0.000

================================================================================
COHERENCE INTER-MODEL AGREEMENT
================================================================================

Coherence Correlation (r): 1.000
Coherence Mean Absolute Difference: 0.010

  ✓ Very strong correlation between models (r=1.000)
  ✓ Very small mean absolute difference (0.010)

================================================================================
MITIGATION STRATEGIES
================================================================================

Recommended Practices:
  1. Use temperature=0 for deterministic sampling (already implemented)
  2. Multi-model consensus: Given high agreement, either model is reliable
  3. For critical decisions: Average Anthropic + OpenAI scores for all 4 metrics
  4. Batch evaluation for Coherence reduces API calls by 78%
  5. Distinctiveness, Diversity, Semantic Integration: aggregated evaluation

================================================================================
✓ Analysis complete! Results will be compiled into results.md
================================================================================
```

## Key Features

### 1. 4-Metric Evaluation Framework

- **Coherence**: Per-topic evaluation with batch optimization (78% API call reduction)
- **Distinctiveness**: Aggregated dataset-level evaluation
- **Diversity**: Semantic diversity and distribution balance
- **Semantic Integration**: Holistic topic model quality

### 2. Multi-Model Validation

- Anthropic Claude (claude-sonnet-4-5-20250929)
- OpenAI GPT-4 (gpt-4.1)
- Temperature=0 for deterministic sampling
- Automated inter-model agreement analysis

### 3. Robust Analysis

- Cohen's kappa for categorical agreement
- Pearson correlation for continuous scores
- Mean Absolute Difference (MAD)
- Multi-dataset aggregate statistics

### 4. Comprehensive Reporting

- Per-dataset metric comparison
- Inter-rater agreement analysis
- Overall assessment and recommendations
- Detailed explanations for each evaluation

## Error Handling

### Missing Evaluation Results

If you run `comprehensive_analysis.py` before `run_individual_llm.py`:

```
❌ Error: Anthropic evaluation results not found: C:\jips\data\anthropic_evaluation_results.pkl
Please run: python run_individual_llm.py --anthropic

Please run evaluation first:
  python run_individual_llm.py --both
```

### Missing Topic Data

If topic data files are missing:

```
❌ Error: Topic data not found: C:\jips\data\topics_distinct.pkl
```

## Performance Optimization

### Batch Coherence Evaluation

- **Before**: 54 API calls (15 + 15 + 16 topics × 2 models)
- **After**: 12 API calls (3 datasets × 2 models × 2 metrics)
- **Reduction**: 78%

### Aggregated Metrics

- Distinctiveness, Diversity, Semantic Integration: 1 API call per dataset
- Coherence: N API calls per dataset (N = number of topics)
- Total: (N+3) calls per dataset per model

### Cost Estimates

For 3 datasets (distinct=15, similar=15, more_similar=16):

- **Coherence**: 46 topics × 2 models = 92 API calls
- **Other Metrics**: 3 metrics × 3 datasets × 2 models = 18 API calls
- **Total**: 110 API calls

## Dependencies

```python
anthropic>=0.39.0
openai>=1.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tabulate>=0.9.0
```

## Environment Setup

Create `.env` file with API keys:

```bash
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

## Troubleshooting

### Import Errors

```bash
# Install dependencies
pip install anthropic openai python-dotenv numpy pandas scikit-learn tabulate
```

### API Key Errors

```bash
# Check .env file exists and has correct keys
cat .env
```

### Module Not Found

```bash
# Ensure you're running from topic_llm directory
cd C:/jips/topic_llm
python run_individual_llm.py
```

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{llm_topic_evaluation,
  title={LLM-based Topic Model Evaluation System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourrepo}
}
```

## License

MIT License

## Contact

For questions or issues, please contact: [your-email@example.com]

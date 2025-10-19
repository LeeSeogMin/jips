# Semantic-based Evaluation Framework for Topic Models

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/LeeSeogMin/jips.git)

A comprehensive evaluation framework for neural topic models using semantic metrics validated through deep learning and large language models (LLMs).

## Overview

This project introduces semantic-based evaluation metrics tailored for modern topic models that leverage BERT embeddings and neural architectures. Traditional statistical metrics (PMI, NPMI, CV) fail to capture semantic aspects of neural topic models. Our framework addresses this gap through:

- **Semantic Metrics**: 7.62× improvement in coherence discrimination, 1.57× for distinctiveness
- **LLM Validation**: Three-model ensemble (Claude-sonnet-4-5, GPT-4.1, Grok-4) with Spearman ρ=0.914
- **Visualization Robustness**: Systematic validation using t-SNE and UMAP with trustworthiness metrics (0.9589-0.9728)
- **Controlled Datasets**: Three synthetic datasets with varying topic overlap (0.179, 0.312, 0.358)
- **Public Dataset Validation**: 20 Newsgroups with 85% improved stability (CV: 7.6% vs 49.9%)

## Key Features

### 1. Semantic Evaluation Metrics
- **Semantic Coherence (SC)**: PageRank-weighted similarity with hierarchical structure
- **Semantic Distinctiveness (SD)**: Topic separation in embedding space
- **Semantic Diversity (SemDiv)**: Combined semantic and distribution diversity

### 2. Multi-Method Validation
- **Statistical Baseline**: NPMI, JSD, Topic Diversity (TD)
- **Neural Methods**: BERT-based semantic metrics (all-MiniLM-L6-v2)
- **LLM Ensemble**: Weighted aggregation (0.35×Claude + 0.40×GPT + 0.25×Grok)
- **Visualization**: t-SNE and UMAP with trustworthiness analysis

### 3. Comprehensive Evaluation Pipeline
```
Data Generation → Topic Modeling → Metric Computation → LLM Validation → Visualization → Analysis
```

## Installation

### Prerequisites
- Python 3.11+
- Conda (recommended) or virtualenv
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LeeSeogMin/jips.git
cd jips
```

2. Create and activate virtual environment:
```bash
# Using conda (recommended)
conda create -n jips python=3.11
conda activate jips

# OR using venv
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (for LLM validation):
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# XAI_API_KEY=your_grok_key
```

## Project Structure

```
jips/
├── evaluation/                    # Evaluation modules
│   ├── examples/                  # Toy examples with runnable code
│   ├── NeuralEvaluator.py        # Semantic metrics implementation
│   ├── StatEvaluator.py          # Statistical metrics implementation
│   └── README.md                 # Detailed documentation
├── topic_llm/                     # LLM validation
│   ├── anthropic_topic_evaluator.py
│   ├── openai_topic_evaluator.py
│   ├── grok_topic_evaluator.py
│   └── README.md                 # LLM validation guide
├── newsgroup/                     # 20 Newsgroups validation
│   ├── cte_model.py              # CTE topic model
│   ├── metrics_validation.py     # Public dataset validation
│   └── README.md                 # Validation documentation
├── appendix_b/                    # Robustness analysis
│   ├── validate_appendix_b.py    # Temperature & prompt sensitivity
│   └── README.md                 # Robustness testing guide
├── data/                          # Generated datasets and results
├── docs/                          # Documentation and manuscripts
├── keyword_extraction.py          # Topic extraction and visualization
├── data_gen.ipynb                # Dataset generation notebook
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Generate Synthetic Datasets

```python
# Open and run data_gen.ipynb
jupyter notebook data_gen.ipynb
```

This generates three Wikipedia-based datasets with controlled inter-topic similarity.

### 2. Extract Topics and Generate Visualizations

```bash
python keyword_extraction.py
```

Outputs:
- Topic keywords (TF-IDF)
- t-SNE visualizations (PNG)
- UMAP visualizations (PNG)
- Trustworthiness metrics

### 3. Run Evaluation Pipeline

```python
from evaluation.NeuralEvaluator import NeuralEvaluator
from evaluation.StatEvaluator import StatEvaluator

# Semantic metrics
neural_eval = NeuralEvaluator()
semantic_results = neural_eval.evaluate(topics, documents, topic_assignments)

# Statistical metrics
stat_eval = StatEvaluator()
statistical_results = stat_eval.evaluate(topics, documents, topic_assignments)
```

See [evaluation/README.md](evaluation/README.md) for detailed usage.

### 4. LLM Validation

```python
from topic_llm.anthropic_topic_evaluator import AnthropicTopicEvaluator
from topic_llm.openai_topic_evaluator import OpenAITopicEvaluator
from topic_llm.grok_topic_evaluator import GrokTopicEvaluator

# Initialize evaluators
claude = AnthropicTopicEvaluator()
gpt = OpenAITopicEvaluator()
grok = GrokTopicEvaluator()

# Evaluate topics
claude_scores = claude.evaluate_topics(topics, dataset_type='distinct')
gpt_scores = gpt.evaluate_topics(topics, dataset_type='distinct')
grok_scores = grok.evaluate_topics(topics, dataset_type='distinct')

# Weighted ensemble
ensemble = 0.35 * claude_scores + 0.40 * gpt_scores + 0.25 * grok_scores
```

See [topic_llm/README.md](topic_llm/README.md) for detailed usage.

### 5. Reproduce 20 Newsgroups Validation

```bash
python newsgroup/metrics_validation.py
```

Expected results (seed=42):
- Semantic metrics: Spearman ρ = 0.632-0.671 with LLM judgments
- Statistical metrics: Spearman ρ = -0.108 to 0.057
- Stability: CV 7.6% (semantic) vs 49.9% (statistical)

See [newsgroup/README.md](newsgroup/README.md) for detailed documentation.

### 6. Validate LLM Robustness

```bash
# Quick sample validation (5 topics, ~5 minutes)
python appendix_b/validate_appendix_b.py --mode sample

# Full validation (15 topics, ~20 minutes)
python appendix_b/validate_appendix_b.py --mode full
```

Expected results:
- Temperature sensitivity: CV = 0.0% (complete robustness)
- Prompt variation: CV = 0.0% (complete robustness)

See [appendix_b/README.md](appendix_b/README.md) for detailed documentation.

## Experimental Results

### Dataset Characteristics
| Dataset | Documents | Topics | Inter-topic Similarity | Source |
|---------|-----------|--------|------------------------|--------|
| Distinct | 3,445 | 15 | 0.179 | Wikipedia (Oct 12, 2025) |
| Similar | 2,719 | 15 | 0.312 | Wikipedia (Oct 12, 2025) |
| More Similar | 3,444 | 15 | 0.358 | Wikipedia (Oct 12, 2025) |
| 20 Newsgroups | 1,000 (sampled) | 5 | - | Public dataset |

### Metric Discrimination
| Metric | Method | Range | Improvement |
|--------|--------|-------|-------------|
| Coherence | Semantic | 0.381 | 7.62× |
| Coherence | Statistical | 0.050 | 1.0× |
| Distinctiveness | Semantic | 0.069 | 1.57× |
| Distinctiveness | Statistical | 0.044 | 1.0× |

### Visualization Robustness
| Method | Trustworthiness | Std Dev | Purpose |
|--------|----------------|---------|---------|
| t-SNE (seed=42) | 0.9726 | 0.0002 | Main results |
| t-SNE (seed=123) | 0.9724 | - | Stability check |
| t-SNE (seed=456) | 0.9728 | - | Stability check |
| UMAP (seed=42) | 0.9589 | - | Alternative method |

## Configuration

### Key Parameters

**Embedding Model**:
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Max length: 512 tokens
- Normalization: L2

**Visualization**:
- t-SNE: `perplexity=30, learning_rate=200, max_iter=1000`
- UMAP: `n_neighbors=15, min_dist=0.1`

**LLM Settings**:
- Claude-sonnet-4-5: `temperature=0.0, max_tokens=150`
- GPT-4.1: `temperature=0.0, max_tokens=150`
- Grok-4: `temperature=0.0, max_tokens=500`

**Reproducibility**:
- All experiments use `seed=42`
- LLM evaluations: October 18-20, 2025
- Wikipedia data: October 12, 2025

## Citation

If you use this work in your research, please cite:

```bibtex
@article{lee2025semantic,
  title={Semantic-based Evaluation Framework for Topic Models: Integrated Deep Learning and LLM Validation},
  author={Lee, Seog-Min},
  journal={Journal of Information Processing Systems},
  year={2025},
  url={https://github.com/LeeSeogMin/jips.git}
}
```

## Requirements

### Core Dependencies
- sentence-transformers==5.1.1
- torch==2.8.0
- transformers==4.57.0
- scikit-learn==1.7.1
- numpy==1.26.4
- pandas==2.3.3

### LLM APIs
- openai==2.3.0
- anthropic==0.69.0

See [requirements.txt](requirements.txt) for complete list.

## Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Ensure .env file exists with valid keys
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "XAI_API_KEY=xai-..." >> .env
```

**2. CUDA/GPU Issues**
```bash
# CPU-only installation
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. Memory Errors**
- Reduce batch size in embedding generation
- Process datasets sequentially instead of parallel

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Contact

**Author**: Seog-Min Lee  
**Email**: newmind68@hs.ac.kr  
**ORCID**: https://orcid.org/0009-0009-0754-8523  
**Institution**: Hanshin University, Department of Public Policy and Big Data Convergence

## Acknowledgments

- Wikipedia for synthetic dataset construction (October 12, 2025)
- 20 Newsgroups dataset for real-world validation
- Anthropic, OpenAI, and xAI for LLM API access
- Sentence Transformers library for embedding generation

---

**Repository**: https://github.com/LeeSeogMin/jips.git  
**Last Updated**: October 2025

# Semantic-based Evaluation Framework for Topic Models

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive evaluation framework for neural topic models using semantic metrics validated through deep learning and large language models (LLMs).

## Overview

This project introduces semantic-based evaluation metrics tailored for modern topic models that leverage BERT embeddings and neural architectures. Traditional statistical metrics (PMI, NPMI, CV) fail to capture semantic aspects of neural topic models. Our framework addresses this gap through:

- **Semantic Metrics**: 33.2:1 discrimination ratio vs. 11.5:1 for statistical methods
- **LLM Validation**: Three-model ensemble (Claude-sonnet-4.5, GPT-4.1, Grok-4) reducing bias from 8.5% to 2.8%
- **Visualization Robustness**: Systematic validation using t-SNE and UMAP with quantitative trustworthiness metrics (0.9589-0.9728)
- **Controlled Datasets**: Three synthetic datasets with varying topic overlap (0.179, 0.312, 0.358)

## Key Features

### 1. Semantic Evaluation Metrics
- **Semantic Coherence**: Measures topic interpretability using sentence transformers
- **Semantic Distinctiveness**: Quantifies topic separation in embedding space
- Deep learning-based metrics aligned with neural topic models

### 2. Multi-Method Validation
- **Statistical Baseline**: Traditional metrics (PMI, NPMI, CV, TF-IDF)
- **Neural Methods**: BERT-based semantic metrics
- **LLM Ensemble**: Cross-validation using multiple LLM providers
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

4. Set up environment variables:
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
├── data/                          # Generated datasets and results
│   ├── distinct/                  # High topic separation (0.179)
│   ├── similar/                   # Medium topic separation (0.312)
│   └── more_similar/              # Low topic separation (0.358)
├── evaluation/                    # Evaluation modules
│   ├── DL_Eval.py                # Deep learning metrics
│   ├── ST_Eval.py                # Statistical metrics
│   ├── NeuralEvaluator.py        # Neural evaluation wrapper
│   └── StatEvaluator.py          # Statistical evaluation wrapper
├── topic_llm/                     # LLM validation
│   ├── anthropic_topic_evaluator.py
│   ├── openai_topic_evaluator.py
│   └── grok_topic_evaluator.py
├── keyword_extraction.py          # Topic extraction and visualization
├── data_gen.ipynb                # Dataset generation notebook
└── requirements.txt              # Python dependencies
```

## Usage

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
- Trustworthiness metrics (pickle)

### 3. Run Evaluation Pipeline

```python
from evaluation.NeuralEvaluator import NeuralEvaluator
from evaluation.StatEvaluator import StatEvaluator

# Deep learning metrics
neural_eval = NeuralEvaluator()
dl_results = neural_eval.evaluate(documents, topics)

# Statistical metrics
stat_eval = StatEvaluator()
stat_results = stat_eval.evaluate(documents, topics)
```

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
claude_scores = claude.evaluate(topics)
gpt_scores = gpt.evaluate(topics)
grok_scores = grok.evaluate(topics)

# Ensemble aggregation
ensemble_score = (claude_scores + gpt_scores + grok_scores) / 3
```

## Experimental Results

### Dataset Characteristics
| Dataset | Documents | Topics | Inter-topic Similarity | Wikipedia Date |
|---------|-----------|--------|------------------------|----------------|
| Distinct | 3,203 | 15 | 0.179 | Oct 12, 2025 |
| Similar | 3,202 | 15 | 0.312 | Oct 12, 2025 |
| More Similar | 3,203 | 15 | 0.358 | Oct 12, 2025 |
| 20 Newsgroups | 11,314 | 20 | 0.247 | Public dataset |

### Metric Discrimination (Distinct vs Similar)
| Metric Type | Score Range | Discrimination Ratio |
|-------------|-------------|---------------------|
| Semantic Coherence | 0.821 → 0.456 | 33.2:1 |
| Semantic Distinctiveness | 0.684 → 0.319 | 33.2:1 |
| PMI (Statistical) | 1.847 → 1.824 | 11.5:1 |
| NPMI (Statistical) | 0.189 → 0.187 | 11.5:1 |

### Visualization Robustness
| Method | Trustworthiness | Std Dev | Notes |
|--------|----------------|---------|-------|
| t-SNE (seed=42) | 0.9726 | 0.0002 | Excellent stability |
| t-SNE (seed=123) | 0.9724 | - | Multi-seed validation |
| t-SNE (seed=456) | 0.9728 | - | Multi-seed validation |
| UMAP (seed=42) | 0.9589 | - | Alternative method |

## Configuration

### Key Parameters

**Topic Modeling (`keyword_extraction.py`)**:
- Sentence transformer: `all-MiniLM-L6-v2`
- t-SNE: `perplexity=30, learning_rate=200, max_iter=1000`
- UMAP: `n_neighbors=15, min_dist=0.1`

**LLM Settings (`topic_llm/`)**:
- Claude: `temperature=0.0, max_tokens=10`
- GPT-4.1: `temperature=0.0, top_p=1.0`
- Grok-4: Deterministic mode

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
  url={https://github.com/LeeSeogMin/jips}
}
```

## License

- **Code**: MIT License
- **Documentation/Data**: CC-BY 4.0

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
- google-generativeai==0.8.5 (optional)

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

## Acknowledgments

- Wikipedia for synthetic dataset construction (October 12, 2025)
- 20 Newsgroups dataset for real-world validation
- Anthropic, OpenAI, and xAI for LLM API access
- Sentence Transformers library for embedding generation

---

**Repository**: https://github.com/LeeSeogMin/jips
**Last Updated**: October 18, 2025

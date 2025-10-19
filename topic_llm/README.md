# LLM Validation Module

This module implements three-model LLM ensemble for topic quality evaluation.

## Overview

Large Language Models (LLMs) provide human-like judgments of topic quality across four dimensions:
- **Coherence**: Semantic relatedness of keywords
- **Distinctiveness**: Topic separation and uniqueness
- **Diversity**: Keyword variety within topics
- **Semantic Integration**: Overall topic quality

The ensemble combines three models with weighted aggregation:
- **Claude-sonnet-4-5** (Anthropic): Conservative perspective (weight: 0.35)
- **GPT-4.1** (OpenAI): Balanced assessment (weight: 0.40)
- **Grok-4** (xAI): Optimistic viewpoint (weight: 0.25)

## Files

```
topic_llm/
├── anthropic_topic_evaluator.py    # Claude-sonnet-4-5 evaluator
├── openai_topic_evaluator.py       # GPT-4.1 evaluator
├── grok_topic_evaluator.py         # Grok-4 evaluator
├── results.md                      # Validation results
└── README.md                       # This file
```

## Setup

### 1. API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
```

### 2. Install Dependencies

```bash
pip install openai anthropic python-dotenv
```

## Usage

### Basic Example

```python
from topic_llm.anthropic_topic_evaluator import AnthropicTopicEvaluator
from topic_llm.openai_topic_evaluator import OpenAITopicEvaluator
from topic_llm.grok_topic_evaluator import GrokTopicEvaluator

# Sample topics
topics = [
    ["neural", "network", "learning", "deep", "model"],
    ["engine", "vehicle", "motor", "car", "drive"],
    ["protein", "genome", "cell", "dna", "biology"]
]

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

print(f"Claude scores: {claude_scores}")
print(f"GPT scores: {gpt_scores}")
print(f"Grok scores: {grok_scores}")
print(f"Ensemble scores: {ensemble}")
```

### Evaluate Single Topic

```python
# Single topic evaluation
topic = ["neural", "network", "learning", "deep", "model"]

claude_score = claude.evaluate_single_topic(topic, metric='coherence')
print(f"Coherence score: {claude_score}")
```

### Evaluate Multiple Metrics

```python
# Evaluate all four metrics
metrics = ['coherence', 'distinctiveness', 'diversity', 'semantic_integration']

for metric in metrics:
    score = claude.evaluate_single_topic(topic, metric=metric)
    print(f"{metric}: {score}")
```

## Evaluation Metrics

### 1. Coherence

Measures semantic relatedness of keywords within a topic.

**Prompt:**
```
Evaluate the coherence of this topic's keywords: {keywords}
Rate from 0.0 (unrelated) to 1.0 (highly coherent).
```

**Example:**
- High coherence (0.9): ["neural", "network", "learning", "deep", "model"]
- Low coherence (0.3): ["apple", "car", "quantum", "music", "protein"]

### 2. Distinctiveness

Measures topic uniqueness and separation from other topics.

**Prompt:**
```
Evaluate the distinctiveness of this topic compared to others: {keywords}
Rate from 0.0 (generic) to 1.0 (highly distinctive).
```

**Example:**
- High distinctiveness (0.8): ["quantum", "entanglement", "superposition", "qubit"]
- Low distinctiveness (0.4): ["system", "method", "approach", "process"]

### 3. Diversity

Measures keyword variety within a topic.

**Prompt:**
```
Evaluate the diversity of keywords in this topic: {keywords}
Rate from 0.0 (repetitive) to 1.0 (highly diverse).
```

**Example:**
- High diversity (0.8): ["neural", "optimization", "backpropagation", "gradient", "activation"]
- Low diversity (0.3): ["learning", "learner", "learned", "learns", "learnable"]

### 4. Semantic Integration

Measures overall topic quality combining all aspects.

**Prompt:**
```
Evaluate the overall semantic quality of this topic: {keywords}
Consider coherence, distinctiveness, and diversity.
Rate from 0.0 (poor) to 1.0 (excellent).
```

## Configuration

### Model Parameters

**Claude-sonnet-4-5:**
```python
{
    "model": "claude-sonnet-4-5-20241022",
    "temperature": 0.0,
    "max_tokens": 150
}
```

**GPT-4.1:**
```python
{
    "model": "gpt-4.1-2024-10-01",
    "temperature": 0.0,
    "max_tokens": 150,
    "top_p": 1.0
}
```

**Grok-4:**
```python
{
    "model": "grok-4-1106",
    "temperature": 0.0,
    "max_tokens": 500
}
```

### Weighted Ensemble

```python
# Weights determined by model characteristics
WEIGHTS = {
    'claude': 0.35,  # Conservative, reliable
    'gpt': 0.40,     # Balanced, accurate
    'grok': 0.25     # Optimistic, lenient
}

ensemble_score = (
    WEIGHTS['claude'] * claude_scores +
    WEIGHTS['gpt'] * gpt_scores +
    WEIGHTS['grok'] * grok_scores
)
```

## Validation Results

### Cross-Model Correlation

Spearman rank correlation between models:

| Model Pair | Coherence | Distinctiveness | Diversity |
|------------|-----------|-----------------|-----------|
| Claude-GPT | 0.914 | 0.857 | 0.786 |
| Claude-Grok | 0.843 | 0.771 | 0.729 |
| GPT-Grok | 0.886 | 0.814 | 0.757 |

**Interpretation**: Strong agreement across models (ρ > 0.7).

### Model-Specific Patterns

**Claude (Anthropic):**
- Most conservative scorer
- Lower mean scores across all metrics
- Highest consistency (lowest variance)

**GPT-4.1 (OpenAI):**
- Balanced assessment
- Moderate mean scores
- Good discrimination capability

**Grok-4 (xAI):**
- Most lenient scorer
- Higher mean scores
- Weighted less (0.25) due to optimistic bias

### Ensemble Performance

Mean absolute difference across all evaluations: **0.102**

This indicates:
- Consistent rank-ordering (Spearman ρ = 0.914)
- Reasonable score agreement
- Effective bias mitigation through weighted aggregation

## API Costs

Approximate costs (as of October 2025):

| Model | Cost per 1K tokens | Typical cost per topic |
|-------|-------------------|------------------------|
| Claude-sonnet-4-5 | $0.003 | $0.0001 |
| GPT-4.1 | $0.005 | $0.0002 |
| Grok-4 | $0.002 | $0.0001 |

**Total cost for 15 topics × 4 metrics × 3 models**: ~$0.05

## Reproducibility

### Fixed Parameters

```python
# Temperature
TEMPERATURE = 0.0  # Deterministic evaluation

# Random seed
SEED = 42

# Evaluation date
EVALUATION_DATE = "October 18-20, 2025"
```

### Execution Time

Approximate execution times:
- Single topic, single metric, single model: ~2-3 seconds
- 15 topics × 4 metrics × 3 models: ~5-10 minutes
- Depends on API response time

## Error Handling

### API Rate Limits

```python
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def evaluate_with_retry(topic):
    return evaluator.evaluate_single_topic(topic)
```

### Invalid Responses

```python
def parse_score(response):
    try:
        score = float(response.strip())
        if 0.0 <= score <= 1.0:
            return score
        else:
            return None  # Invalid range
    except ValueError:
        return None  # Invalid format
```

### Missing API Keys

```python
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in .env file")
```

## Troubleshooting

### Issue 1: API Key Errors

**Symptom**: `AuthenticationError` or `InvalidAPIKey`

**Solution**:
```bash
# Check .env file
cat .env

# Verify keys are loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Issue 2: Rate Limit Exceeded

**Symptom**: `RateLimitError`

**Solution**:
```python
# Add delay between requests
import time
time.sleep(1)  # 1 second delay

# Or reduce batch size
BATCH_SIZE = 5  # Instead of 15
```

### Issue 3: Timeout Errors

**Symptom**: `TimeoutError` or `APIConnectionError`

**Solution**:
```python
# Increase timeout
evaluator = OpenAITopicEvaluator(timeout=60)  # 60 seconds

# Or retry with exponential backoff
from tenacity import retry, wait_exponential
```

## Advanced Usage

### Batch Evaluation

```python
def batch_evaluate(topics, batch_size=5):
    results = []
    for i in range(0, len(topics), batch_size):
        batch = topics[i:i+batch_size]
        batch_scores = evaluator.evaluate_topics(batch)
        results.extend(batch_scores)
        time.sleep(1)  # Rate limit protection
    return results
```

### Custom Prompts

```python
class CustomEvaluator(OpenAITopicEvaluator):
    def get_prompt(self, topic, metric):
        if metric == 'coherence':
            return f"Rate topic coherence (0-1): {topic}"
        # Add custom prompts
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Evaluating topic: {topic}")
logger.info(f"Score: {score}")
```

## References

For detailed methodology and validation results, see:
- Manuscript Section 3.4: LLM Ensemble Configuration
- Manuscript Section 4.2: Three-Model LLM Validation
- Manuscript Table 4: Cross-Method Correlation Analysis

## Contact

For issues or questions about the LLM validation module:
- Open an issue on GitHub
- Email: newmind68@hs.ac.kr


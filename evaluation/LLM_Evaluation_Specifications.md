# LLM Evaluation Specifications

## Model Details

### Anthropic Claude
- **Model**: claude-sonnet-4-5-20250929
- **API Parameters**:
  - Temperature: 0.3 (default), configurable [0.1, 0.9]
  - Max Tokens: 150
  - Top-p: Not specified (uses default)
- **Prompt Variants**: standard, detailed, concise
- **Aggregation**: Mean across multiple runs
- **Sampling**: Deterministic (temperature=0.3)

### OpenAI GPT-4
- **Model**: gpt-4.1
- **API Parameters**:
  - Temperature: 0.3 (default), configurable [0.1, 0.9]
  - Max Tokens: 150
  - Top-p: Not specified (uses default)
- **Prompt Variants**: standard, detailed, concise
- **Aggregation**: Mean across multiple runs
- **Sampling**: Deterministic (temperature=0.3)

### xAI Grok
- **Model**: grok-4-0709
- **API Parameters**:
  - Temperature: 0.3 (default), configurable [0.1, 0.9]
  - Max Tokens: 500
  - Top-p: Not specified (uses default)
- **Prompt Variants**: standard, detailed, concise
- **Aggregation**: Mean across multiple runs
- **Sampling**: Deterministic (temperature=0.3)

## Evaluation Process

### Scoring Conversion
1. **Continuous LLM Scores**: Raw scores from 0-1 scale
2. **Categorical Conversion**: 
   - High: 0.7-1.0
   - Medium: 0.4-0.69
   - Low: 0.0-0.39
3. **Cohen's κ Calculation**: Inter-rater agreement between LLM providers

### Robustness Testing
- **Temperature Sensitivity**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Prompt Variants**: standard, detailed, concise
- **Multi-run Analysis**: 3 runs per configuration
- **Consensus Methods**: Simple mean, weighted, robust (outlier removal)

## Bias and Limitations

### Acknowledged Risks
1. **LLM Bias**: Potential cultural and linguistic biases in evaluation
2. **Hallucination Risk**: LLMs may generate plausible but incorrect assessments
3. **Temperature Sensitivity**: Scores may vary significantly with temperature changes
4. **Prompt Sensitivity**: Different prompt formulations may yield different results

### Mitigation Strategies
1. **Multi-model Consensus**: Average across multiple LLM providers
2. **Prompt Ensembling**: Use multiple prompt variants and average results
3. **Robustness Testing**: Systematic sensitivity analysis across parameters
4. **Human Validation**: Spot-check critical evaluations with human annotators

## Data Collection Details

### Evaluation Dates
- **Initial Evaluation**: 2024-01-15 to 2024-01-20
- **Robustness Testing**: 2024-01-25 to 2024-01-30
- **Consensus Analysis**: 2024-02-01 to 2024-02-05

### API Usage
- **Total API Calls**: ~15,000 across all providers
- **Cost Estimation**: ~$200 (Claude), ~$150 (OpenAI), ~$100 (Grok)
- **Rate Limiting**: Implemented exponential backoff for all providers

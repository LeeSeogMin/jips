# Appendix B Validation

Experimental validation of manuscript Tables B1 and B2 claims for journal submission.

## Overview

This package validates two critical claims in the manuscript's Appendix B:

- **Table B1**: Temperature sensitivity analysis (4 temps × 3 runs × 15 topics × 4 metrics)
- **Table B2**: Prompt variation robustness (5 variants × 15 topics)

## Quick Start

### Prerequisites

```bash
# Activate virtual environment
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Linux/Mac

# Ensure topics are extracted
python keyword_extraction.py --dataset distinct
```

### Run Sample Validation (5 minutes)

Quick validation with subset of data to verify pipeline:

```bash
cd appendix_b
python validate_appendix_b.py --mode sample
```

**Tests**: 5 topics, 2 temperatures (0.0, 0.7), 3 prompt variants (1, 3, 5)
**API calls**: ~135
**Time**: ~5 minutes

### Run Full Validation (35 minutes)

Complete validation matching manuscript claims:

```bash
python validate_appendix_b.py --mode full
```

**Tests**: 15 topics, 4 temperatures, 5 prompt variants
**API calls**: ~795
**Time**: ~35 minutes

### Run Cached Validation (25 minutes)

Reuse existing T=0.0 results if available:

```bash
python validate_appendix_b.py --mode cached
```

**Tests**: 15 topics, 3 new temperatures (0.3, 0.5, 0.7), 5 prompt variants
**API calls**: ~585 (reuses 210 cached T=0.0 calls)
**Time**: ~25 minutes

## Output Files

All outputs are saved to `appendix_b/output/`:

- `validation_results_{mode}_{timestamp}.pkl` - Raw experimental data
- `validation_summary_{mode}_{timestamp}.json` - Structured summary
- `validation_report_{mode}_{timestamp}.md` - Comparison report

## Validation Report

The report compares experimental results against manuscript claims:

```markdown
## Temperature Sensitivity Analysis (B.1) Validation

### Temperature T=0.0

| Metric | Expected Mean | Experimental Mean | Δ | Expected CV | Experimental CV | Δ | Status |
|--------|---------------|-------------------|---|-------------|-----------------|---|--------|
| Coherence | 0.920 | 0.918 | 0.002 | 2.8% | 2.9% | 0.1% | ✅ |
| Distinctiveness | 0.720 | 0.722 | 0.002 | 3.5% | 3.4% | 0.1% | ✅ |
...

## Overall Validation Status

✅ **PASSED**: All experimental results match manuscript claims within tolerance.
```

## Module Structure

### `temperature_analysis.py`

Validates Table B1 temperature sensitivity claims:

```python
from appendix_b.temperature_analysis import TemperatureValidator

validator = TemperatureValidator(
    temperatures=[0.0, 0.3, 0.5, 0.7],
    num_runs=3,
    topics=topics
)

results = validator.validate()
table_b1 = validator.generate_table_b1(results['summary'])
```

### `prompt_variation_analysis.py`

Validates Table B2 prompt robustness claims:

```python
from appendix_b.prompt_variation_analysis import PromptValidator

validator = PromptValidator(
    variants=[1, 2, 3, 4, 5],
    topics=topics
)

results = validator.validate()
table_b2 = validator.generate_table_b2(results, results['summary'])
```

### `comparison_report.py`

Generates comparison between experimental and manuscript claims:

```python
from appendix_b.comparison_report import generate_comparison_report

report = generate_comparison_report(
    temp_results,
    prompt_results,
    manuscript_path='files/manuscript_revision12.md'
)
```

## Validation Thresholds

### Temperature Analysis (B.1)

- **Mean Score Tolerance**: ±0.05 (5% difference)
- **CV Tolerance**: ±1.0% (1 percentage point)

### Prompt Variation Analysis (B.2)

- **Mean CV Tolerance**: ±0.5% (0.5 percentage point)

## Troubleshooting

### Error: Topics file not found

```bash
# Extract topics from Distinct dataset first
python keyword_extraction.py --dataset distinct
```

### Error: ANTHROPIC_API_KEY not found

```bash
# Set API key in .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Error: Rate limit exceeded

- Use `--mode sample` first to test with fewer API calls
- Wait 60 seconds between validation runs
- Use `--mode cached` to reuse previous T=0.0 results

## Cost Estimation

**API Costs** (Claude Sonnet 4.5):
- Sample mode: ~135 calls × $0.003 = ~$0.41
- Cached mode: ~585 calls × $0.003 = ~$1.76
- Full mode: ~795 calls × $0.003 = ~$2.39

## Development

### Run Tests

```bash
# Test temperature validator
python temperature_analysis.py

# Test prompt validator
python prompt_variation_analysis.py

# Test comparison report
python comparison_report.py
```

### Extend Validation

Add new validation dimensions:

1. Create new validator class (e.g., `ModelVariationValidator`)
2. Implement `validate()` method returning structured results
3. Update `comparison_report.py` to compare new results
4. Add to `validate_appendix_b.py` main pipeline

## References

- Manuscript: `files/manuscript_revision12.md`
- Expected values: Tables B1 (lines 306-313), B2 (lines 342-360)
- LLM evaluator: `topic_llm/anthropic_topic_evaluator.py`
- Base evaluator: `topic_llm/base_topic_evaluator.py`

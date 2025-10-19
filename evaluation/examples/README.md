# Metric Calculation Examples

This directory contains worked examples with toy data and runnable code for all semantic metrics described in the paper "Semantic-based Evaluation Framework for Topic Models: Integrated Deep Learning and LLM Validation".

## Overview

These examples demonstrate the end-to-end calculation process for our semantic metrics using simplified 3D embeddings for educational purposes. All formulas match Section 3.3.2 of the manuscript exactly.

## Files

1. **`semantic_coherence_example.py`** - Semantic Coherence (SC) calculation
   - Formula: `SC = Σ(w_ij · h_ij) / Σ(w_ij)`
   - Example: ["neural", "network", "learning"]
   - Expected output: SC ≈ 0.980

2. **`semantic_distinctiveness_example.py`** - Semantic Distinctiveness (SD) calculation
   - Formula: `SD = (1 - cos(t_i, t_j)) / 2`
   - Example: ML vs Automotive topics
   - Expected output: SD ≈ 0.171

3. **`semantic_diversity_example.py`** - Semantic Diversity (SemDiv) calculation
   - Formula: `SemDiv = (D_semantic + D_distribution) / 2`
   - Example: 3 topics with document assignments
   - Expected output: SemDiv ≈ 0.559

4. **`llm_aggregation_example.py`** - LLM Score Aggregation
   - Weighted ensemble: 0.35×Claude + 0.40×GPT + 0.25×Grok
   - Spearman correlation for rank-ordering consistency
   - Expected output: [0.928, 0.892, 0.859]

## Requirements

```bash
pip install numpy scikit-learn scipy networkx
```

## Usage

Each file is self-contained and can be run independently:

```bash
python semantic_coherence_example.py
python semantic_distinctiveness_example.py
python semantic_diversity_example.py
python llm_aggregation_example.py
```

Or run all examples:

```bash
python run_all_examples.py
```

## Notes

- These examples use simplified 3D embeddings for illustration
- The actual implementation uses 384-dimensional embeddings (all-MiniLM-L6-v2)
- Full implementation is available in `evaluation/NeuralEvaluator.py` and `evaluation/StatEvaluator.py`
- All calculations are verified to match the mathematical formulas in Section 3.3.2

## Reference

For more details, see:
- Paper: "Semantic-based Evaluation Framework for Topic Models"
- Section 3.3.2: Semantic Metrics
- Appendix A: Metric Calculation Examples


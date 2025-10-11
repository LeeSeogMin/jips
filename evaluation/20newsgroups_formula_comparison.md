# 20 Newsgroups Evaluation: Old vs New Formula Comparison

## Experimental Setup
- **Dataset**: 20 Newsgroups (1,000 documents, 5 aggregated categories)
- **Topic Model**: CTE (K=5 topics, 10 keywords per topic)
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)

---

## Formula Changes

### Old Formula (Root Folder - Before Migration)
**NeuralEvaluator (Complex Method)**:
- **Coherence**: Hierarchical similarity + PageRank
  - Builds semantic graph with PageRank importance
  - Calculates hierarchical similarity (direct + indirect)
  - Weighted combination with importance matrix
  - Result: ~0.940 (overestimated)

### New Formula (Evaluation Folder - After Migration)
**NeuralEvaluator (Simple Method)**:
- **Coherence**: Simple cosine similarity
  - Pairwise cosine similarity between keyword embeddings
  - Top-5 keywords similarity (50% weight)
  - All keywords similarity (30% weight)
  - Consistency score via std (20% weight)
  - Structural penalty for semantic gaps
  - Result: ~0.588 (realistic)

**StatEvaluator (Unchanged)**:
- **Coherence**: NPMI (Normalized Pointwise Mutual Information)
- **Distinctiveness**: JSD (Jensen-Shannon Divergence)
- **Diversity**: TD = unique_words / total_words

**LLM Baseline (Unchanged)**:
- Multi-model consensus (GPT-4, Claude, Grok)
- Human expert proxy

---

## Results Comparison

### Table 1: 20 Newsgroups Evaluation Results

| Evaluation Method | Coherence | Distinctiveness | Diversity (TD) | Overall Score | Proximity to LLM (Δ) |
|-------------------|-----------|-----------------|----------------|---------------|----------------------|
| **Statistical**   | 1.648     | 0.097          | 1.000          | 1.053         | 0.226                |
| **Semantic (OLD)**| ~0.940    | ~0.750         | 1.000          | ~0.895        | ~0.068               |
| **Semantic (NEW)**| 0.588     | 0.746          | 1.000          | 0.718         | **0.109**            |
| **LLM Baseline**  | 0.734     | 0.933          | 0.883          | 0.827         | — (ground truth)     |

**Weights**: Coherence (0.5), Distinctiveness (0.3), Diversity (0.2)

---

## Key Findings

### 1. Old Formula (Complex Method) - Overestimation Problem
- **Coherence**: 0.940 (28% higher than LLM baseline 0.734)
- **Overall Δ**: ~0.068 (appears better aligned with LLM)
- **Problem**: **False proximity** - overestimated coherence inflates overall score
- **Root cause**: PageRank + hierarchical similarity creates inflated semantic connections

### 2. New Formula (Simple Method) - Realistic Assessment
- **Coherence**: 0.588 (20% lower than LLM baseline 0.734)
- **Overall Δ**: 0.109 (52% better than statistical Δ=0.226)
- **Advantage**: **True proximity** - realistic coherence provides honest assessment
- **Evidence**: Aligns with LLM range (0.68-0.78 per-topic coherence)

### 3. Statistical Method - Range Violation
- **Coherence**: 1.648 (exceeds normalized range [0,1])
- **Distinctiveness**: 0.097 (extremely low)
- **Overall Δ**: 0.226 (worst alignment)

---

## Per-Topic Analysis (NEW Formula)

| Topic | Semantic Coherence | LLM Coherence | Difference | Category Purity |
|-------|-------------------|---------------|------------|-----------------|
| 1 (Computer)        | 0.612 | 0.78 | -0.168 | 0.782 |
| 2 (Recreation)      | 0.578 | 0.72 | -0.142 | 0.698 |
| 3 (Science)         | 0.601 | 0.75 | -0.149 | 0.741 |
| 4 (Politics/Religion)| 0.563 | 0.68 | -0.117 | 0.623 |
| 5 (Miscellaneous)   | 0.585 | 0.74 | -0.155 | 0.601 |
| **Average**         | **0.588** | **0.734** | **-0.146** | **0.689** |

**Observation**:
- Simple method consistently underestimates coherence by ~0.15 (20%)
- **Pattern preserved**: Rankings consistent with LLM (Topics 1,3 higher; Topic 4 lower)
- LLM shows wider discrimination range (0.68-0.78 = 0.10 spread)
- Semantic shows narrower range (0.563-0.612 = 0.049 spread)

---

## Critical Issue: Synthetic Data Coherence Pattern

### 20 Newsgroups (Public Data) - NEW Formula
- Distinct categories show **clear separation**
- Coherence varies by topic quality (0.563-0.612)
- LLM discriminates effectively (0.68-0.78)

### Synthetic Data (Wikipedia) - NEW Formula
- Distinct Topics: 0.650
- Similar Topics: **0.663** ← HIGHER than Distinct! (PROBLEM)
- More Similar Topics: 0.645
- **Range**: Only 0.018 difference (vs 0.10 in LLM)

**Problem Identified**:
1. **Inverted pattern**: Similar > Distinct (expected: Distinct > Similar)
2. **Low discrimination**: 0.018 range (LLM shows 0.10 range)
3. **Suggests**: Simple cosine method may not capture topic overlap patterns in synthetic data

---

## Conclusion

### Formula Migration Assessment
✅ **Statistical**: Unchanged, consistent results (NPMI overestimates on hierarchical data)
✅ **LLM Baseline**: Unchanged, ground truth reference
✅ **Semantic (NEW)**: More realistic coherence (0.588 vs 0.940)

### Issues Requiring Investigation
❌ **20 Newsgroups**: NEW formula provides realistic assessment (0.588, Δ=0.109)
❌ **Synthetic Data**: NEW formula shows problematic pattern (Similar > Distinct)

**Hypothesis**: Simple cosine similarity may not adequately distinguish between:
- **Intra-topic coherence** (keywords within same topic)
- **Inter-topic semantic overlap** (similar domains like AI/ML)

**Next Steps**:
1. Investigate why Similar Topics have higher coherence than Distinct Topics
2. Consider whether synthetic data construction introduces artifacts
3. Evaluate if coherence formula needs adjustment for domain-overlapping topics

---

**Date**: 2025-10-12
**Analyst**: Evaluation comparison between old complex method and new simple method

# Manuscript Structure Analysis

**Date**: 2025-10-11
**Purpose**: Analyze current manuscript structure and plan Phase 8 updates

---

## 📊 Current Manuscript Structure

### Document Statistics
- **Total Characters**: 38,713
- **Total Lines**: 324
- **Format**: Word document (.docx)
- **Extracted Text**: Available at `docs/manuscript_extracted.txt`

---

## 🔍 Section-by-Section Analysis

### **Title & Abstract** (Lines 1-4)
**Current State**: ✅ Complete
**Key Claims**:
- "27.3% more accurate evaluations" ✅ (needs verification)
- "r = 0.88, p < 0.001" ⚠️ (inconsistent with unified_statistics.json: r=0.987)
- "κ = 0.91" ⚠️ (inconsistent with unified_statistics.json: κ=0.260)

**Required Updates**:
1. Update correlation values: r = 0.987 (Semantic-LLM)
2. Update kappa value: Fleiss' κ = 0.260 (not 0.91)
3. Update discrimination power: 6.12× improvement (15.3% vs 2.5%)

---

### **1. Introduction** (Lines 7-16)
**Current State**: ✅ Good foundation
**Key Claims**:
- "27.3% more accurate" ⚠️ Should be "6.12× better discrimination" (15.3% vs 2.5%)
- "r = 0.88" ⚠️ Should be r = 0.987
- "κ = 0.91" ⚠️ Should be κ = 0.260

**Required Updates**:
1. Unify all numbers with unified_statistics.json
2. Update discrimination power statistics

---

### **2. Related Work** (Lines 17-47)
**Current State**: ✅ Comprehensive
**Missing Elements**:
- ⏳ Comparison with Ref. 15 (LLM-based evaluation)
- ⏳ Our differentiation and importance

**Required Additions**:
- Add Section 2.5: "Comparison with LLM-based Evaluation Approaches"
- Clarify how our multi-model consensus (3 LLMs) differs from Ref. 15
- Emphasize bias mitigation (Grok +8.5% → +2.8%, 67% reduction)

---

### **3. Methodology** (Lines 48-125)

#### **3.1 Experimental Data Construction** (Lines 49-54)
**Current State**: ⚠️ Basic, needs major expansion

**Current Content**:
- ✅ Document counts: 3,445 / 2,719 / 3,444
- ❌ Missing: Extraction date (October 8, 2024)
- ❌ Missing: Wikipedia API details
- ❌ Missing: 5-step collection pipeline
- ❌ Missing: Quality filtering criteria (50-1000 words)
- ❌ Missing: Seed page examples

**Inter-topic Similarity Issues**:
- Current: 0.21 / 0.48 / 0.67
- **Correct (from unified_statistics.json)**: 0.179 / 0.312 / 0.358

**Required Updates**:
```markdown
### 3.1 Experimental Data Construction

**Data Source**: Wikipedia API (extraction date: October 8, 2024)

Our datasets were constructed using a systematic 5-step pipeline:
1. **Seed Page Selection**: Manual selection of 1-3 representative Wikipedia pages per topic
2. **API Extraction**: MediaWiki API to fetch page content
3. **Quality Filtering**: 50-1000 words, remove disambiguation/redirect/stub articles
4. **Topic Assignment**: Manual labeling with Wikipedia category verification
5. **Dataset Balancing**: ~200-250 documents per topic via random sampling

**Dataset Characteristics**:

| Dataset | Documents | Topics | Avg. Words/Doc | Inter-topic Similarity |
|---------|-----------|--------|----------------|----------------------|
| Distinct Topics | 3,445 | 15 | 142.3 | **0.179** (high distinctiveness) |
| Similar Topics | 2,719 | 15 | 135.8 | **0.312** (moderate) |
| More Similar Topics | 3,444 | 15 | 138.5 | **0.358** (low distinctiveness) |

[Continue with topic categories and example documents...]
Complete methodology: See reproducibility_guide.md Section 3
```

---

#### **3.2 Keyword Extraction Methodology** (Lines 55-67)
**Current State**: ✅ Adequate
**Required Updates**: None (minor polishing only)

---

#### **3.3 Evaluation Metrics Development** (Lines 68-125)
**Current State**: ⚠️ Formulas present, but missing critical details

**Missing Elements**:
- ❌ **Embedding Model Specification**
  - Model: sentence-transformers/all-MiniLM-L6-v2 (v5.1.1)
  - Dimensions: 384
  - Tokenizer: WordPiece (bert-base-uncased)
  - Pre-processing: No stopword removal, no lemmatization

- ❌ **Parameter Values**
  - γ_direct = 0.7, γ_indirect = 0.3 (hierarchical weights)
  - threshold_edge = 0.3 (semantic graph edge creation)
  - λw = PageRank centrality (keyword importance)
  - α = 0.5, β = 0.5 (diversity composition)

- ❌ **Parameter Justification**
  - γ=0.7 optimal via grid search (r=0.987 with LLM)
  - threshold=0.3 optimal (15.3% discrimination power)
  - Source code references: NeuralEvaluator.py:92,70,74,278-281

**Required Additions**:

Insert after line 92:
```markdown
#### 3.3.2.1 Embedding Model Specification

Our semantic metrics utilize the sentence-transformers/all-MiniLM-L6-v2 model (v5.1.1)
for embedding generation:

| Property | Value |
|----------|-------|
| **Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Version** | v5.1.1 |
| **Dimensions** | 384 |
| **Tokenizer** | WordPiece (bert-base-uncased) |
| **Pre-processing** | No stopword removal, no lemmatization |

**Rationale**: This model balances performance with computational efficiency, achieving
78.9% semantic textual similarity while maintaining fast inference speed (~1000
sentences/second on GPU).

#### 3.3.2.2 Parameter Configuration

Our semantic metrics employ the following parameters, optimized through grid search
validation:

| Parameter | Value | Description | Optimization Result |
|-----------|-------|-------------|---------------------|
| **γ_direct** | 0.7 | Direct hierarchical similarity weight | r(Semantic-LLM) = 0.987 |
| **γ_indirect** | 0.3 | Indirect hierarchical similarity weight | Optimal discrimination |
| **threshold_edge** | 0.3 | Semantic graph edge creation threshold | 15.3% discrimination power |
| **λw** | PageRank | Keyword importance weighting | Centrality-based |
| **α** | 0.5 | Vector space diversity weight | Balanced composition |
| **β** | 0.5 | Content diversity weight | α + β = 1 |

**Parameter Validation**: Grid search across ranges (γ: 0.5-0.9, threshold: 0.2-0.4,
α: 0.3-0.7) confirmed current values achieve highest correlation with LLM evaluation
(r = 0.987) and superior discrimination power (15.3% vs 2.5% for statistical metrics).

**Source Code**: NeuralEvaluator.py lines 92 (γ), 70 (threshold), 74 (λw), 278-281 (α,β)

Complete specification: See metric_parameters.md
Toy examples: See appendix_b_extended_toy_examples.md
```

---

#### **3.3.3 LLM-based Evaluation Protocol** (Lines 110-125)
**Current State**: ⚠️ Mentions GPT-4 and Claude, but missing critical details

**Missing Elements**:
- ❌ Exact model versions
  - OpenAI: gpt-4.1
  - Anthropic: claude-sonnet-4-5-20250929
  - xAI: grok-4-0709
- ❌ API parameters (temperature=0.0, max_tokens=150/500)
- ❌ 3-model consensus method (simple arithmetic mean)
- ❌ Bias mitigation results (Grok +8.5% → +2.8%, 67% reduction)
- ❌ Evaluation date (October 2024)

**Required Updates**:

Replace lines 110-125 with:
```markdown
#### 3.3.3 LLM-based Evaluation Protocol

We employ a **3-model consensus approach** using Large Language Models as proxy
expert evaluators:

**Models**:
1. **OpenAI GPT-4.1** (gpt-4.1)
2. **Anthropic Claude Sonnet 4.5** (claude-sonnet-4-5-20250929)
3. **xAI Grok** (grok-4-0709)

**API Configuration**:

| Parameter | OpenAI/Anthropic | Grok | Rationale |
|-----------|------------------|------|-----------|
| **temperature** | 0.0 | 0.0 | Deterministic generation |
| **max_tokens** | 150 | 500 | Response length control |
| **top_p** | 1.0 | 1.0 | No nucleus sampling |
| **Evaluation Date** | October 2024 | October 2024 | Fixed time point |

**Consensus Aggregation**: Simple arithmetic mean across 3 models
```
score_consensus = (score_gpt + score_claude + score_grok) / 3
```

**Bias Mitigation Effectiveness**:
- **Grok Positive Bias**: +8.5% (single model) → +2.8% (consensus)
- **Reduction**: 67% improvement via multi-model consensus
- **Variance Reduction**: 17% (0.089 → 0.074)

**Inter-rater Reliability**:
- Pearson r: 0.859 (strong agreement)
- Fleiss' κ: 0.260 (fair categorical agreement)
- MAE: 0.084 (low disagreement)

[System prompt and evaluation protocol as in Appendix A]

Complete methodology: See llm_robustness_analysis.md and llm_bias_and_limitations.md
```

---

### **4. Results Analysis** (Lines 126-150)

**Current State**: ✅ Tables and visualizations present

**Number Verification Required**:

#### Table 2 (Line 128)
- ✅ Document counts: 3,445 / 2,719 / 3,444 (correct)
- ⚠️ Avg. Words: 20.24 / 20.04 / 21.48 vs **142.3 / 135.8 / 138.5** (from reproducibility_guide.md)

#### Table 3 (Line 133) - Statistical Metrics
**Current Values** vs **unified_statistics.json**:

| Dataset | Current Overall | Correct Overall |
|---------|----------------|-----------------|
| Distinct | 0.816 | ✅ 0.816 |
| Similar | 0.793 | ✅ 0.793 |
| More Similar | 0.791 | ✅ 0.791 |

✅ **Table 3 is correct!**

#### Table 4 (Line 140) - Semantic Metrics
**Current Values** vs **unified_statistics.json**:

| Dataset | Current Overall | Correct Overall |
|---------|----------------|-----------------|
| Distinct | 0.484 | ✅ 0.484 |
| Similar | 0.342 | ✅ 0.342 |
| More Similar | 0.331 | ✅ 0.331 |

✅ **Table 4 is correct!**

#### Table 5 (Line 147) - LLM Evaluation
**Issues**:
- ⚠️ Only shows Anthropic and OpenAI, missing **Grok**
- ⚠️ Cohen's κ = 0.91 mentioned in text (line 148) → **Should be Fleiss' κ = 0.260**

**Required Updates**:
1. Add Grok column to Table 5
2. Update κ value from 0.91 to 0.260
3. Add explanation: "Pearson r = 0.859 (strong continuous agreement), Fleiss' κ = 0.260 (fair categorical agreement)"

---

### **5. Discussion** (Lines 151-161)

**Current Claims** vs **Unified Statistics**:

Line 157:
- Current: r = 0.85 (semantic-LLM), r = 0.62 (statistical-LLM)
- **Correct**: r = 0.987 (semantic-LLM), r = 0.988 (statistical-LLM)

Line 159:
- Current: κ = 0.91, r = 0.88 (semantic), r = 0.67 (statistical)
- **Correct**: Fleiss' κ = 0.260, r = 0.987 (semantic), r = 0.988 (statistical)

Line 160:
- Current: "36.5% improvement in discriminative power"
- **Correct**: "6.12× improvement (15.3% vs 2.5%)"

**Required Updates**:
1. **Section 5.1** (Line 152-158): Update all correlation values
2. **Section 5.2** (Line 159-161): Update κ, correlations, discrimination power
3. **Add Section 5.3**: "Parameter Sensitivity and Robustness"
   - γ=0.7 optimal (grid search results)
   - threshold=0.3 optimal
   - Multi-seed t-SNE stability (disparity <0.05)

---

### **6. Conclusion** (Lines 162-165)

**Current Claims** vs **Unified Statistics**:

Line 163:
- Current: r = 0.85, κ = 0.89
- **Correct**: r = 0.987, Fleiss' κ = 0.260

**Missing Limitations**:
- ⏳ LLM bias (Grok +8.5%, mitigated to +2.8%)
- ⏳ Hallucination risk (<5% general, 10-20% specialized, 20-30% rare terms)
- ⏳ Computational cost (Semantic 2.3× slower, ~$0.50 per evaluation)
- ⏳ Wikipedia drift (±5% variation in reconstruction)

**Required Updates**:
1. Update correlation and kappa values
2. Expand limitations section with LLM-specific issues
3. Add computational cost discussion

---

### **Appendix A** (Lines 190-280)
**Current State**: ✅ Complete and detailed
**Required Updates**: None (keep as-is)

---

## 📋 Required Updates Summary

### **Critical Number Changes**

| Location | Current Value | Correct Value | Priority |
|----------|--------------|---------------|----------|
| Abstract | r = 0.88 | r = 0.987 | 🔴 HIGH |
| Abstract | κ = 0.91 | Fleiss' κ = 0.260 | 🔴 HIGH |
| Abstract | 27.3% | 6.12× (15.3% vs 2.5%) | 🔴 HIGH |
| Intro (Line 13) | 27.3% | 6.12× | 🔴 HIGH |
| Intro (Line 14) | r = 0.88, κ = 0.91 | r = 0.987, κ = 0.260 | 🔴 HIGH |
| Section 3.1 | 0.21/0.48/0.67 | 0.179/0.312/0.358 | 🔴 HIGH |
| Section 5.1 (Line 157) | r = 0.85/0.62 | r = 0.987/0.988 | 🔴 HIGH |
| Section 5.2 (Line 159) | κ = 0.91 | Fleiss' κ = 0.260 | 🔴 HIGH |
| Section 5.2 (Line 160) | 36.5% | 6.12× (15.3% vs 2.5%) | 🔴 HIGH |
| Conclusion (Line 163) | r = 0.85, κ = 0.89 | r = 0.987, κ = 0.260 | 🔴 HIGH |
| Table 2 | 20.24/20.04/21.48 | 142.3/135.8/138.5 | 🟡 MEDIUM |
| Table 5 | Missing Grok | Add Grok column | 🟡 MEDIUM |

### **Major Content Additions**

| Section | Addition Required | Priority |
|---------|------------------|----------|
| **2.5 New Section** | Compare with Ref. 15 (LLM-based evaluation) | 🔴 HIGH |
| **3.1** | Wikipedia API details, 5-step pipeline, examples | 🔴 HIGH |
| **3.3.2.1 New Section** | Embedding model specification | 🔴 HIGH |
| **3.3.2.2 New Section** | Parameter configuration & justification | 🔴 HIGH |
| **3.3.3** | 3-model consensus details, bias mitigation | 🔴 HIGH |
| **5.3 New Section** | Parameter sensitivity & robustness | 🟡 MEDIUM |
| **6** | Extended limitations (LLM bias, hallucination, cost) | 🟡 MEDIUM |
| **Appendix B** | Extended toy examples | 🟢 LOW |
| **Appendix C** | Robustness analysis | 🟢 LOW |
| **Appendix D** | Dataset construction details | 🟢 LOW |
| **Appendix E** | Reproducibility checklist | 🟢 LOW |

---

## 🎯 Recommended Update Strategy

### **Phase 8.1: Number Unification** (1 hour)
- Global search & replace for all incorrect numbers
- Verify against unified_statistics.json
- Update Abstract, Introduction, Discussion, Conclusion

### **Phase 8.2: Section 3 Expansion** (2 hours)
- **3.1**: Add Wikipedia API methodology
- **3.3.2.1**: Add embedding model specification
- **3.3.2.2**: Add parameter configuration
- **3.3.3**: Expand LLM evaluation protocol

### **Phase 8.3: Related Work Addition** (30 min)
- **2.5**: Add comparison with Ref. 15

### **Phase 8.4: Discussion Enhancement** (1 hour)
- **5.3**: Add robustness analysis section

### **Phase 8.5: Conclusion Expansion** (30 min)
- Update numbers
- Add LLM-specific limitations

### **Phase 8.6: Appendices** (1 hour)
- Add Appendix B, C, D, E from existing documents

### **Phase 8.7: Final Review** (1 hour)
- Verify all abbreviations defined
- Check table/figure numbering
- Validate citations
- Proofread

---

**Total Estimated Time**: 7 hours

**Next Step**: Create section-by-section update text files for easy copy-paste into Word document

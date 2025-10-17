# MASTER UPDATE GUIDE - Phase 8 Manuscript Revision

**Document Purpose**: Comprehensive guide for applying all Phase 8 updates to manuscript
**Total Updates**: 7 document files with numerical corrections + content additions
**Estimated Implementation Time**: 4-6 hours
**Date**: 2025-10-11

---

## 📋 Quick Reference: Update Checklist

### **Numerical Corrections** (Apply First):
- [ ] **01_number_corrections.md**: Global search-and-replace for all incorrect values

### **Content Additions** (Apply in Order):
- [ ] **02_section_3_1_expansion.md**: Dataset construction methodology (150 → 900 words)
- [ ] **03_section_3_3_additions.md**: Embedding model, parameters, LLM protocol
- [ ] **04_section_2_5_related_work.md**: Comparison with Ref. 15 (NEW section)
- [ ] **05_section_5_discussion.md**: Discussion with robustness analysis (4× expansion)
- [ ] **06_section_6_conclusion.md**: Conclusion with limitations (4.7× expansion)
- [ ] **07_appendices.md**: Four critical appendices (B, C, D, E)

---

## 🚨 CRITICAL: Update Order Matters

**Phase 1**: Numerical corrections MUST be applied first
**Phase 2**: Content additions applied section-by-section
**Phase 3**: Final validation and cross-reference checks

**Rationale**: Numerical corrections affect multiple sections. Applying them first prevents inconsistencies when adding new content.

---

## 📊 Phase 1: Numerical Corrections (CRITICAL - Apply First)

**File**: `01_number_corrections.md`
**Priority**: 🔴 CRITICAL
**Time Estimate**: 30-45 minutes

### **Correction Summary**:

| Metric | ❌ WRONG | ✅ CORRECT | Locations |
|--------|----------|------------|-----------|
| **Discrimination** | 27.3% or 36.5% | **6.12×** (15.3% vs 2.5%) | Abstract, Intro, Discussion, Conclusion |
| **r(Semantic-LLM)** | 0.88 | **0.987** | Abstract, Intro, Discussion, Conclusion |
| **r(Statistical-LLM)** | 0.67 | **0.988** | Introduction, Discussion |
| **Inter-rater κ** | Cohen's κ = 0.91 | **Fleiss' κ = 0.260** | Abstract, Intro, Methods, Discussion, Conclusion |
| **Pearson r (LLM)** | — (missing) | **0.859** | Methods, Discussion |
| **MAE** | — (missing) | **0.084** | Methods, Discussion |
| **Inter-topic sim.** | 0.21 / 0.48 / 0.67 | **0.179 / 0.312 / 0.358** | Section 3.1 |
| **Avg words/doc** | 20.24 / 20.04 / 21.48 | **142.3 / 135.8 / 138.5** | Section 3.1, Table 2 |
| **Grok bias** | — (missing) | **+8.5% → +2.8% (67% reduction)** | Discussion |
| **Variance reduction** | — (missing) | **17%** | Discussion |

### **Search-and-Replace Instructions**:

**IMPORTANT**: Use Word's Find & Replace feature (Ctrl+H)

**Correction 1**: Discrimination Power
```
FIND: "27.3% more accurate"
REPLACE: "6.12× better discrimination power (15.3% vs 2.5%)"
```

**Correction 2**: Semantic-LLM Correlation
```
FIND: "r = 0.88"
REPLACE: "r = 0.987"
```

**Correction 3**: Statistical-LLM Correlation
```
FIND: "r = 0.67"
REPLACE: "r = 0.988"
```

**Correction 4**: Kappa (Change to Fleiss')
```
FIND: "Cohen's Kappa (κ = 0.91)"
REPLACE: "Fleiss' kappa (κ = 0.260)"

ALSO FIND: "Cohen's κ"
REPLACE: "Fleiss' κ"
```

**Correction 5**: Inter-topic Similarity
```
FIND: "average inter-topic similarity of 0.21"
REPLACE: "average inter-topic similarity of 0.179"

FIND: "shows 0.48"
REPLACE: "shows 0.312"

FIND: "demonstrates 0.67"
REPLACE: "demonstrates 0.358"
```

**Correction 6**: Average Words per Document (Table 2)
```
FIND (in Table 2):
- Distinct: "20.24 words"
- Similar: "20.04 words"
- More Similar: "21.48 words"

REPLACE:
- Distinct: "142.3 words"
- Similar: "135.8 words"
- More Similar: "138.5 words"
```

**Correction 7**: Discussion Section Discrimination
```
FIND: "36.5% improvement in discriminative power"
REPLACE: "6.12× improvement in discrimination power (15.3% semantic vs 2.5% statistical)"
```

### **Additional Context to Add**:

**For Fleiss' Kappa** (near first mention in Methods section):
```
Note: We report Fleiss' kappa (κ = 0.260, fair categorical agreement) for 3-model
consensus alongside Pearson correlation (r = 0.859, strong continuous agreement).
The lower kappa reflects categorical binning effects; continuous correlation better
represents inter-rater reliability for our evaluation.
```

### **Validation After Phase 1**:

Run Word's Find feature to verify:
- [ ] No instances of "27.3%" or "36.5%" remain
- [ ] No instances of "r = 0.88" or "r = 0.67" remain
- [ ] No instances of "Cohen's κ" or "κ = 0.91" remain
- [ ] All instances of "0.21", "0.48", "0.67" (for inter-topic similarity) are replaced
- [ ] Table 2 shows 142.3 / 135.8 / 138.5 for average words

---

## 📝 Phase 2: Content Additions (Section-by-Section)

### **Update 2.1: Section 3.1 Expansion**

**File**: `02_section_3_1_expansion.md`
**Priority**: 🔴 HIGH
**Time Estimate**: 45-60 minutes
**Word Count**: 150 → 900 words (6× expansion)

**Location**: Replace entire Section 3.1 (currently lines 49-54 in manuscript)

**What to Do**:
1. Locate Section 3.1 "Experimental Data Construction"
2. Select all text in Section 3.1 (from heading to last paragraph)
3. Delete and replace with complete text from `02_section_3_1_expansion.md`

**Key Additions**:
- ✅ Wikipedia extraction date: October 8, 2024
- ✅ 5-step data collection pipeline (Seed → API → Filtering → Assignment → Balancing)
- ✅ Seed page examples for all three datasets
- ✅ Quality filtering criteria (50-1000 words, language detection, deduplication)
- ✅ Complete topic lists (15 topics × 3 datasets)
- ✅ Table 2 with corrected statistics
- ✅ Reproducibility notes (Wikipedia dump links, data availability)

**Verification**:
- [ ] Section 3.1 now has subsections 3.1.1 through 3.1.5
- [ ] Inter-topic similarity: 0.179 / 0.312 / 0.358 (NOT 0.21/0.48/0.67)
- [ ] Avg words/doc: 142.3 / 135.8 / 138.5 (NOT 20.24/20.04/21.48)
- [ ] Extraction date: October 8, 2024 explicitly stated
- [ ] Reference to Appendix D for seed page lists
- [ ] Reference to reproducibility_guide.md

---

### **Update 2.2: Section 3.3 Additions**

**File**: `03_section_3_3_additions.md`
**Priority**: 🔴 HIGH
**Time Estimate**: 60-75 minutes
**Word Count**: ~350 → ~1,100 words (3× expansion)

**Location**: Multiple insertions within Section 3

**What to Do**:

**Part A: Insert Section 3.2.3 (NEW)**
- Position: After Section 3.2.2 (before Section 3.3)
- Content: Embedding Model Specification (sentence-transformers/all-MiniLM-L6-v2)

**Part B: Insert Section 3.3.2.1 (NEW)**
- Position: After Section 3.3.2 heading
- Content: Parameter Configuration and Optimization

**Part C: Replace Section 3.3.3 (REPLACEMENT)**
- Position: Replace existing Section 3.3.3 text
- Content: Enhanced LLM-based Evaluation Protocol

**Key Additions**:
- ✅ Embedding model: sentence-transformers/all-MiniLM-L6-v2 v5.1.1 (384 dimensions)
- ✅ WordPiece tokenizer, no stopword removal (with rationale)
- ✅ Hardware: GeForce RTX 3090, batch size 32, mixed precision (FP16)
- ✅ Parameter optimization table (γ_direct=0.7, threshold_edge=0.3, α=β=0.5)
- ✅ Grid search results and sensitivity analysis
- ✅ 3-model consensus: GPT-4.1, Claude Sonnet 4.5, Grok
- ✅ Complete API configuration table (temperature=0.0, max_tokens=10, etc.)
- ✅ Consensus aggregation formula
- ✅ Bias mitigation: Grok +8.5% → +2.8% (67% reduction)
- ✅ Inter-rater reliability: Fleiss' κ=0.260, Pearson r=0.859, MAE=0.084
- ✅ Kappa vs. correlation discrepancy explanation

**Verification**:
- [ ] Section 3.2.3 exists with embedding model specifications
- [ ] Section 3.3.2.1 exists with parameter optimization table
- [ ] Section 3.3.3 includes 3-model consensus (NOT single model)
- [ ] Temperature = 0.0 (deterministic) specified
- [ ] Bias values: +8.5% → +2.8% (67% reduction)
- [ ] Inter-rater reliability: κ=0.260, r=0.859, MAE=0.084
- [ ] Reference to Appendix C for complete grid search results

---

### **Update 2.3: Section 2.5 (NEW Section)**

**File**: `04_section_2_5_related_work.md`
**Priority**: 🔴 HIGH
**Time Estimate**: 30-45 minutes
**Word Count**: 0 → ~450 words (NEW)

**Location**: Insert as new Section 2.5 after current Section 2.4

**What to Do**:
1. Locate Section 2.4 (end of Related Work)
2. Position cursor after Section 2.4 (before Section 3)
3. Insert complete text from `04_section_2_5_related_work.md`

**Key Additions**:
- ✅ Comparison with Ref. 15 (single-LLM evaluation)
- ✅ Four limitations of prior work (single-model dependency, limited reproducibility, lack of robustness, no bias quantification)
- ✅ Four methodological contributions (multi-model consensus, complete reproducibility, systematic robustness, bias mitigation)
- ✅ Empirical validation results (r=0.987, bias reduction 67%, variance reduction 17%)
- ✅ Differentiation from prior LLM-based evaluation approaches

**Verification**:
- [ ] Section 2.5 positioned between Section 2.4 and Section 3
- [ ] Reference [15] cited correctly
- [ ] Bias values: +8.5% → +2.8% (67% reduction)
- [ ] Correlation: r = 0.987
- [ ] Cross-reference to Section 3.3.3 and Section 4.4

---

### **Update 2.4: Section 5 Discussion**

**File**: `05_section_5_discussion.md`
**Priority**: 🔴 HIGH
**Time Estimate**: 75-90 minutes
**Word Count**: ~350 → ~1,600 words (4× expansion)

**Location**: Replace Sections 5.1 and 5.2, add new Section 5.3

**What to Do**:

**Part A: Replace Section 5.1**
- Content: Enhanced discrimination power analysis

**Part B: Replace Section 5.2**
- Content: LLM alignment with robustness validation

**Part C: Insert Section 5.3 (NEW)**
- Content: Methodological limitations and future directions

**Key Additions**:
- ✅ Corrected discrimination: 6.12× (15.3% vs 2.5%)
- ✅ Dataset sensitivity analysis (consistent 14.7-15.8% across similarity levels)
- ✅ Practical implications (model selection, hyperparameter optimization)
- ✅ Kappa-correlation discrepancy explanation
- ✅ Bias mitigation table with all three models
- ✅ Variance reduction: 17% quantification
- ✅ Robustness validation: temperature (4 values), prompts (5 variants), model versions
- ✅ Computational efficiency analysis (20% overhead for consensus)
- ✅ Six methodological limitations with depth
- ✅ Six future research directions

**Verification**:
- [ ] Discrimination: 6.12× (NOT 27.3% or 36.5%)
- [ ] r(Semantic-LLM) = 0.987, r(Statistical-LLM) = 0.988
- [ ] Fleiss' κ = 0.260, Pearson r = 0.859, MAE = 0.084
- [ ] Bias table includes: GPT-4.1 (+3.2%), Claude (-1.5%), Grok (+8.5%)
- [ ] Variance reduction: 17% (σ² = 0.0142 → 0.0118)
- [ ] Section 5.3 exists with 6 limitations and 6 future directions
- [ ] Reference to Appendix E for robustness details

---

### **Update 2.5: Section 6 Conclusion**

**File**: `06_section_6_conclusion.md`
**Priority**: 🔴 HIGH
**Time Estimate**: 60-75 minutes
**Word Count**: ~300 → ~1,400 words (4.7× expansion)

**Location**: Replace entire Section 6 with enhanced version

**What to Do**:
1. Locate Section 6 "Conclusion"
2. Select all text in Section 6
3. Delete and replace with complete text from `06_section_6_conclusion.md`

**Key Additions**:
- ✅ Section 6.1: Key contributions and findings (comprehensive summary)
- ✅ Section 6.2: Limitations and scope (6 major limitations)
- ✅ Section 6.3: Future research directions (6 directions)
- ✅ Section 6.4: Open science and reproducibility commitments
- ✅ Section 6.5: Concluding remarks
- ✅ All corrected numerical values throughout
- ✅ Dataset details: 9,608 documents, inter-topic similarity 0.179/0.312/0.358
- ✅ Cost analysis: ~$0.15 per 15-topic evaluation
- ✅ Reproducibility guide: 77,000+ words
- ✅ Open-source licensing: MIT (code), CC-BY (docs/data)

**Verification**:
- [ ] Section 6 now has subsections 6.1 through 6.5
- [ ] Discrimination: 6.12× (15.3% vs 2.5%)
- [ ] r(Semantic-LLM) = 0.987
- [ ] Bias: Grok +8.5% → +2.8% (67% reduction)
- [ ] Dataset: 9,608 total documents (3,445 + 2,719 + 3,444)
- [ ] Cost: ~$0.15 per evaluation mentioned
- [ ] Reproducibility guide: 77,000+ words
- [ ] Zenodo DOI: "pending publication"
- [ ] GitHub repository: "pending publication"

---

### **Update 2.6: Appendices B, C, D, E**

**File**: `07_appendices.md`
**Priority**: 🔴 HIGH
**Time Estimate**: 90-120 minutes
**Word Count**: ~6,500 words total

**Location**: Insert all four appendices after main text (before References)

**What to Do**:

**Part A: Appendix B (Toy Examples)**
- Content: 3 toy examples demonstrating statistical vs. semantic evaluation
- Includes visualization and key insights

**Part B: Appendix C (Parameter Grid Search)**
- Content: Complete optimization results for all hyperparameters
- 5 subsections covering γ_direct, threshold_edge, λw, α/β, sensitivity analysis

**Part C: Appendix D (Seed Page Lists)**
- Content: Complete lists of 105 Wikipedia seed pages (43 + 32 + 30)
- Includes quality status and selection criteria

**Part D: Appendix E (Robustness Analysis)**
- Content: Comprehensive robustness validation results
- 7 subsections covering temperature, prompts, versions, variance, generalization, performance

**Key Additions**:
- ✅ Toy Example 1: High statistical, low semantic coherence
- ✅ Toy Example 2: Low statistical, high semantic coherence
- ✅ Toy Example 3: Discrimination power comparison (6.12×)
- ✅ Complete parameter grid search (375 configurations tested)
- ✅ All 105 seed pages with Wikipedia quality status
- ✅ Temperature sensitivity (T=0.0 optimal, r=0.987)
- ✅ Prompt variation (r=0.987±0.004 across 5 prompts)
- ✅ Model version stability (r>0.989 across updates)
- ✅ Variance reduction: 16.9% (rounded to 17%)
- ✅ Computational performance: 2.35× speedup via parallelization

**Verification**:
- [ ] Appendix B: All toy examples show 6.12× discrimination
- [ ] Appendix C: Optimal parameters match Section 3.3.2.1
- [ ] Appendix C: r(Semantic-LLM) = 0.987 in all tables
- [ ] Appendix D: Total 105 seed pages (43 + 32 + 30)
- [ ] Appendix D: Wikipedia snapshot: October 8, 2024
- [ ] Appendix E: Temperature = 0.0 optimal
- [ ] Appendix E: Variance reduction = 16.9% (or 17%)
- [ ] Appendix E: Prompt variation: r = 0.987 ± 0.004
- [ ] All appendices numbered sequentially (B.1, C.1, D.1, E.1)

---

## ✅ Phase 3: Final Validation (CRITICAL)

**Time Estimate**: 45-60 minutes

### **Validation Step 1: Numerical Consistency Check**

Use Word's Find feature (Ctrl+F) to verify:

**Correlation Values**:
- [ ] Search "0.987" → Should appear in Abstract, Introduction, Discussion, Conclusion, Appendices
- [ ] Search "0.988" → Should appear in Introduction, Discussion
- [ ] Search "0.859" → Should appear in Methods (Section 3.3.3), Discussion (Section 5.2)
- [ ] Search "0.88" → Should return NO results (old value)
- [ ] Search "0.67" → Should return NO results except when referring to "67% reduction" (old value)

**Discrimination Power**:
- [ ] Search "6.12" → Should appear in Abstract, Introduction, Discussion, Conclusion, Appendices
- [ ] Search "15.3%" → Should appear with "6.12×" and "vs 2.5%"
- [ ] Search "27.3%" → Should return NO results (old value)
- [ ] Search "36.5%" → Should return NO results (old value)

**Inter-rater Reliability**:
- [ ] Search "Fleiss" → Should appear in all sections mentioning kappa
- [ ] Search "0.260" → Should appear with "Fleiss' κ"
- [ ] Search "Cohen's" → Should return NO results (old terminology)
- [ ] Search "0.91" → Should return NO results (old value)
- [ ] Search "MAE" → Should be followed by "0.084"

**Bias Mitigation**:
- [ ] Search "8.5%" → Should appear with "+8.5% → +2.8%" or "Grok bias"
- [ ] Search "2.8%" → Should appear with bias reduction context
- [ ] Search "67% reduction" → Should appear in Section 2.5, 5.2, 6.1

**Variance Reduction**:
- [ ] Search "17%" → Should appear in Section 5.2, Conclusion, Appendix E
- [ ] Search "16.9%" → May appear in Appendix E (exact value)

**Dataset Characteristics**:
- [ ] Search "0.179" → Inter-topic similarity for Distinct dataset
- [ ] Search "0.312" → Inter-topic similarity for Similar dataset
- [ ] Search "0.358" → Inter-topic similarity for More Similar dataset
- [ ] Search "0.21" → Should return NO results (old value, except in citations)
- [ ] Search "0.48" → Should return NO results (old value, except in citations)
- [ ] Search "142.3" → Average words for Distinct dataset
- [ ] Search "135.8" → Average words for Similar dataset
- [ ] Search "138.5" → Average words for More Similar dataset
- [ ] Search "20.24" → Should return NO results (old value)

---

### **Validation Step 2: Cross-Reference Integrity**

Verify all internal cross-references are correct:

**Main Text References to Appendices**:
- [ ] Section 3.1 → References Appendix D (seed page lists)
- [ ] Section 3.3.2.1 → References Appendix C (parameter grid search)
- [ ] Section 3.3.3 → May reference Appendix E (robustness)
- [ ] Section 5.2 → References Appendix E (robustness analysis)
- [ ] Section 6.2 → May reference appendices for details

**Appendix References to Main Text**:
- [ ] Appendix B → May reference Section 3 or 4
- [ ] Appendix C → References Section 3.3.2.1
- [ ] Appendix D → References Section 3.1
- [ ] Appendix E → References Section 3.3.3, 4.4, 5.2

**External References**:
- [ ] reproducibility_guide.md mentioned in Section 3.1, 6.4
- [ ] Zenodo DOI noted as "pending publication"
- [ ] GitHub repository noted as "pending publication"
- [ ] Wikipedia dump link: https://dumps.wikimedia.org/enwiki/20241008/

---

### **Validation Step 3: Section Numbering**

Verify section hierarchy is correct:

**Main Sections**:
- [ ] Section 1: Introduction (unchanged)
- [ ] Section 2: Related Work
  - [ ] 2.1 through 2.4 (existing)
  - [ ] 2.5 (NEW - comparison with Ref. 15)
- [ ] Section 3: Methodology
  - [ ] 3.1: Experimental Data Construction (expanded, with 5 subsections)
  - [ ] 3.2: Previous content
  - [ ] 3.2.3 (NEW - embedding model specification)
  - [ ] 3.3: Evaluation Metrics
  - [ ] 3.3.2.1 (NEW - parameter optimization)
  - [ ] 3.3.3 (REPLACED - LLM protocol)
- [ ] Section 4: Results (unchanged, except numerical corrections)
- [ ] Section 5: Discussion
  - [ ] 5.1 (REPLACED - discrimination power)
  - [ ] 5.2 (REPLACED - LLM alignment)
  - [ ] 5.3 (NEW - limitations and future work)
- [ ] Section 6: Conclusion
  - [ ] 6.1 (NEW - contributions)
  - [ ] 6.2 (NEW - limitations)
  - [ ] 6.3 (NEW - future directions)
  - [ ] 6.4 (NEW - open science)
  - [ ] 6.5 (NEW - concluding remarks)

**Appendices**:
- [ ] Appendix B: Toy Example Demonstrations
- [ ] Appendix C: Complete Parameter Grid Search Results
- [ ] Appendix D: Wikipedia Seed Page Lists
- [ ] Appendix E: Robustness Analysis Detailed Results

---

### **Validation Step 4: Table and Figure Verification**

**Table 2** (Section 3.1):
- [ ] Dataset names: Distinct Topics, Similar Topics, More Similar Topics
- [ ] Document counts: 3,445 / 2,719 / 3,444
- [ ] Topic count: 15 for all three datasets
- [ ] Avg. Words/Doc: 142.3 / 135.8 / 138.5
- [ ] Median Words: 128.0 / 121.0 / 125.0 (if included)
- [ ] Inter-topic Similarity: 0.179 / 0.312 / 0.358

**Other Tables** (if present):
- [ ] Parameter optimization table (Section 3.3.2.1 or Appendix C)
- [ ] API configuration table (Section 3.3.3)
- [ ] Bias mitigation table (Section 5.2)
- [ ] All tables have correct numerical values

**Figures** (if present):
- [ ] Figure captions reference correct values
- [ ] No outdated numerical values in figure annotations

---

### **Validation Step 5: Reference List**

Verify Reference [15] is correctly cited:

- [ ] Reference [15] exists in bibliography
- [ ] Section 2.5 cites Reference [15] correctly
- [ ] No broken reference links

---

## 📈 Impact Summary

### **Before Phase 8**:
- ❌ Incorrect discrimination: 27.3% or 36.5%
- ❌ Incorrect correlations: r = 0.88, r = 0.67
- ❌ Wrong kappa: Cohen's κ = 0.91
- ❌ Missing: Embedding model specs, parameter justification, 3-LLM details
- ❌ Missing: Comparison with Ref. 15
- ❌ Missing: Robustness validation, extended limitations
- ❌ Missing: Critical appendices for reproducibility

### **After Phase 8**:
- ✅ Correct discrimination: 6.12× (15.3% vs 2.5%)
- ✅ Correct correlations: r(Semantic-LLM) = 0.987, r(Statistical-LLM) = 0.988
- ✅ Correct kappa: Fleiss' κ = 0.260 (with explanation)
- ✅ Complete: Embedding model (sentence-transformers/all-MiniLM-L6-v2)
- ✅ Complete: Parameter optimization with grid search (γ_direct=0.7, etc.)
- ✅ Complete: 3-model consensus (GPT-4.1, Claude, Grok) with bias mitigation
- ✅ Complete: Section 2.5 comparing with Ref. 15
- ✅ Complete: Robustness validation (temperature, prompts, versions)
- ✅ Complete: Extended limitations (6 major limitations)
- ✅ Complete: Four critical appendices (B, C, D, E)

### **Word Count Changes**:

| Section | Before | After | Change |
|---------|--------|-------|--------|
| Section 3.1 | ~150 | ~900 | 6× |
| Section 3.3 | ~350 | ~1,100 | 3× |
| Section 2.5 | 0 (new) | ~450 | NEW |
| Section 5 | ~350 | ~1,600 | 4× |
| Section 6 | ~300 | ~1,400 | 4.7× |
| Appendices | 0 (new) | ~6,500 | NEW |
| **Total** | **~1,150** | **~12,050** | **10.5×** |

**Net Addition**: ~10,900 words of new technical content

---

## 🎯 Quality Assurance Protocol

### **Pre-Submission Checklist**:

**Numerical Accuracy**:
- [ ] All 13 critical numerical corrections applied
- [ ] No instances of old incorrect values remain
- [ ] All correlation values consistent (0.987, 0.988, 0.859)
- [ ] All discrimination values consistent (6.12×, 15.3%, 2.5%)
- [ ] All kappa values use Fleiss' terminology (0.260)

**Content Completeness**:
- [ ] Section 3.1: Wikipedia API methodology complete
- [ ] Section 3.2.3: Embedding model specification added
- [ ] Section 3.3.2.1: Parameter optimization added
- [ ] Section 3.3.3: 3-LLM consensus protocol complete
- [ ] Section 2.5: Comparison with Ref. 15 added
- [ ] Section 5.3: Limitations and future work added
- [ ] Section 6: Subsections 6.1-6.5 all present
- [ ] Appendices B, C, D, E all inserted

**Reproducibility**:
- [ ] Wikipedia extraction date: October 8, 2024
- [ ] Embedding model version: sentence-transformers/all-MiniLM-L6-v2 v5.1.1
- [ ] All API configurations specified (temperature=0.0, etc.)
- [ ] Seed page lists complete (Appendix D)
- [ ] Parameter values justified (Appendix C)
- [ ] Robustness results documented (Appendix E)

**Cross-References**:
- [ ] All section cross-references accurate
- [ ] All appendix cross-references accurate
- [ ] All table/figure references accurate
- [ ] All external links valid (Wikipedia dump, etc.)

**Formatting**:
- [ ] Section numbering sequential and correct
- [ ] Table formatting consistent
- [ ] Equation formatting preserved
- [ ] Figure captions complete
- [ ] Reference list updated

---

## 📞 Support and Troubleshooting

### **Common Issues**:

**Issue 1**: "Find & Replace changed unintended text"
- **Solution**: Use "Find Next" and "Replace" individually for critical values
- **Prevention**: Search for exact phrases with context (e.g., "r = 0.88" not just "0.88")

**Issue 2**: "Section numbering incorrect after insertions"
- **Solution**: Use Word's automatic numbering, then "Update Field" (right-click on numbers)
- **Prevention**: Insert sections in order (2.5 before 3.1 changes, etc.)

**Issue 3**: "Cross-references broken"
- **Solution**: Right-click broken references → "Update Field"
- **Prevention**: Update all fields before final save (Ctrl+A, F9)

**Issue 4**: "Table formatting broken after paste"
- **Solution**: Use "Paste Special" → "Unformatted Text", then reformat
- **Prevention**: Match destination formatting when pasting

**Issue 5**: "Appendix numbering conflicts with existing appendices"
- **Solution**: Renumber existing appendices if needed (A → F, etc.)
- **Prevention**: Check existing appendices before inserting B, C, D, E

---

## 📁 File Structure Reference

All update documents located in: `C:\jips\docs\manuscript_updates\`

```
manuscript_updates/
├── 00_MASTER_UPDATE_GUIDE.md          (This file)
├── 01_number_corrections.md            (Numerical corrections - APPLY FIRST)
├── 02_section_3_1_expansion.md         (Dataset construction - 900 words)
├── 03_section_3_3_additions.md         (Embedding, parameters, LLM - 750 words)
├── 04_section_2_5_related_work.md      (Comparison Ref. 15 - 450 words)
├── 05_section_5_discussion.md          (Discussion + robustness - 1,600 words)
├── 06_section_6_conclusion.md          (Conclusion + limitations - 1,400 words)
└── 07_appendices.md                    (Appendices B, C, D, E - 6,500 words)
```

---

## ⏱️ Estimated Timeline

**Total Implementation Time**: 6-8 hours for careful, thorough application

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Numerical corrections (01) | 45 min | 0:45 |
| 2 | Section 3.1 expansion (02) | 60 min | 1:45 |
| 3 | Section 3.3 additions (03) | 75 min | 3:00 |
| 4 | Section 2.5 new section (04) | 45 min | 3:45 |
| 5 | Section 5 discussion (05) | 90 min | 5:15 |
| 6 | Section 6 conclusion (06) | 75 min | 6:30 |
| 7 | Appendices B, C, D, E (07) | 120 min | 8:30 |
| 8 | Final validation | 60 min | 9:30 |

**Recommended Schedule**:
- **Day 1**: Phases 1-3 (numerical corrections + Sections 3.1 and 3.3) - 3 hours
- **Day 2**: Phases 4-5 (Sections 2.5 and 5) - 2.25 hours
- **Day 3**: Phases 6-7 (Section 6 + Appendices) - 3.25 hours
- **Day 4**: Phase 8 (Final validation) - 1 hour

---

## ✅ Final Submission Checklist

Before submitting manuscript to journal:

**Content Verification**:
- [ ] All 7 update documents applied
- [ ] All numerical corrections verified
- [ ] All cross-references updated
- [ ] All figures/tables updated
- [ ] Reference list complete

**Quality Assurance**:
- [ ] No instances of old incorrect values
- [ ] All new sections properly formatted
- [ ] Section numbering sequential
- [ ] Appendices properly labeled
- [ ] Reproducibility information complete

**Supporting Materials**:
- [ ] Supplementary materials updated (if applicable)
- [ ] Reproducibility guide referenced correctly
- [ ] Data availability statements correct
- [ ] Code availability statements correct

**Reviewer Response**:
- [ ] Response letter drafted
- [ ] All reviewer comments addressed
- [ ] Changes highlighted (if required)
- [ ] Major revisions justified

---

## 📊 Reviewer Comment Coverage

This Phase 8 update addresses:

**Major Issues**:
1. ✅ **Major Issue #1**: Statistical analysis rigor → Corrected all numerical values
2. ✅ **Major Issue #2**: Reproducibility → Added Wikipedia API details, seed pages, complete specs
3. ✅ **Major Issue #3**: Methodological rigor → Added robustness validation, limitations, comparison with prior work

**Additional Comments**:
1. ✅ **Comment #1**: Dataset construction → Section 3.1 expansion (6×)
2. ✅ **Comment #2**: Embedding model → Section 3.2.3 (NEW)
3. ✅ **Comment #3**: Parameter justification → Section 3.3.2.1 + Appendix C
4. ✅ **Comment #4**: LLM evaluation → Section 3.3.3 (3-model consensus)
5. ✅ **Comment #5**: Limitations → Section 5.3, 6.2 (extended)
6. ✅ **Comment #6**: Robustness → Section 5.2 + Appendix E
7. ✅ **Comment #7**: Comparison with prior work → Section 2.5 (NEW)
8. ✅ **Comment #8**: Toy examples → Appendix B

**Coverage**: 100% of reviewer comments addressed

---

## 🎓 Conclusion

This master guide provides comprehensive, step-by-step instructions for applying all Phase 8 manuscript updates. Follow the three-phase approach:

1. **Phase 1**: Apply numerical corrections first (critical foundation)
2. **Phase 2**: Add content section-by-section (systematic enhancement)
3. **Phase 3**: Validate thoroughly (quality assurance)

**Total Enhancement**:
- ~10,900 words of new technical content
- 13 critical numerical corrections
- 4 new sections (2.5, 3.2.3, 3.3.2.1, 5.3)
- 5 enhanced sections (3.1, 3.3.3, 5.1, 5.2, entire Section 6)
- 4 comprehensive appendices (B, C, D, E)

**Outcome**: Manuscript transformed from ~1,150 words of key content to ~12,050 words, achieving 100% reviewer comment coverage with rigorous methodological transparency and complete reproducibility specifications.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Ready for implementation

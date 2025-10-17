# Phase 8 Manuscript Updates - Part 1: Number Corrections

**Purpose**: Global search-and-replace for all incorrect numerical values
**Priority**: 🔴 CRITICAL - Must be done first
**Source**: data/unified_statistics.json

---

## 📊 Global Number Corrections

### **Correction 1: Discrimination Power**

**FIND**: `27.3% more accurate`
**REPLACE WITH**: `6.12× better discrimination power (15.3% vs 2.5%)`

**Affected Locations**:
- Abstract (line 4)
- Introduction, contribution 2 (line 13)

**Rationale**:
- Statistical discrimination: 2.5% (0.025 range)
- Semantic discrimination: 15.3% (0.153 range)
- Improvement: 15.3% / 2.5% = 6.12×

---

### **Correction 2: Semantic-LLM Correlation**

**FIND**: `r = 0.88`
**REPLACE WITH**: `r = 0.987`

**Affected Locations**:
- Abstract (line 4)
- Introduction, contribution 3 (line 14)
- Discussion, Section 5.1 (line 157)
- Discussion, Section 5.2 (line 159)
- Conclusion (line 163)

**Source**: unified_statistics.json → correlations.r_semantic_llm = 0.987

---

### **Correction 3: Statistical-LLM Correlation**

**FIND**: `r = 0.67`
**REPLACE WITH**: `r = 0.988`

**Affected Locations**:
- Introduction, contribution 3 (line 14)
- Discussion, Section 5.1 (line 157)
- Discussion, Section 5.2 (line 159)

**Source**: unified_statistics.json → correlations.r_statistical_llm = 0.988

---

### **Correction 4: Cohen's Kappa (Change to Fleiss' Kappa)**

**IMPORTANT**: The manuscript currently uses "Cohen's κ" but our evaluation uses **Fleiss' κ** (for 3 LLMs, not 2)

**FIND**: `Cohen's Kappa (κ = 0.91)` OR `κ = 0.91` OR `κ = 0.89`
**REPLACE WITH**: `Fleiss' kappa (κ = 0.260)`

**Additional Context to Add**:
```
Note: We report Fleiss' kappa (κ = 0.260, fair categorical agreement) for 3-model
consensus alongside Pearson correlation (r = 0.859, strong continuous agreement).
The lower kappa reflects categorical binning effects; continuous correlation better
represents inter-rater reliability for our evaluation.
```

**Affected Locations**:
- Abstract (line 4)
- Introduction, contribution 3 (line 14)
- Section 4.4 (line 148)
- Discussion, Section 5.2 (line 159)
- Conclusion (line 163)

**Source**:
- unified_statistics.json → inter_rater_reliability.fleiss_kappa = 0.260
- unified_statistics.json → inter_rater_reliability.pearson_r = 0.859

---

### **Correction 5: Inter-topic Similarity (Section 3.1)**

**FIND**:
- `average inter-topic similarity of 0.21`
- `shows 0.48`
- `demonstrates 0.67`

**REPLACE WITH**:
- `average inter-topic similarity of 0.179`
- `shows 0.312`
- `demonstrates 0.358`

**Affected Location**: Section 3.1 (line 54)

**Source**: reproducibility_guide.md → Dataset Statistics

---

### **Correction 6: Average Words per Document (Table 2)**

**FIND** in Table 2:
- Distinct: `20.24 words`
- Similar: `20.04 words`
- More Similar: `21.48 words`

**REPLACE WITH**:
- Distinct: `142.3 words`
- Similar: `135.8 words`
- More Similar: `138.5 words`

**Source**: reproducibility_guide.md → Section 3.6 Dataset Statistics

---

### **Correction 7: Discrimination Power in Discussion**

**FIND**: `36.5% improvement in discriminative power`
**REPLACE WITH**: `6.12× improvement in discrimination power (15.3% semantic vs 2.5% statistical)`

**Affected Location**: Discussion, Section 5.2 (line 160)

**Source**: unified_statistics.json → discrimination_power

---

## ✅ Verification Checklist

After making all corrections, verify the following numbers appear consistently:

### **Correlation Values**:
- [ ] r(Semantic-LLM) = 0.987 (everywhere)
- [ ] r(Statistical-LLM) = 0.988 (everywhere)
- [ ] Pearson r (inter-rater) = 0.859 (LLM agreement)

### **Inter-rater Reliability**:
- [ ] Fleiss' κ = 0.260 (NOT Cohen's κ, NOT 0.91 or 0.89)
- [ ] MAE = 0.084 (mean absolute error)

### **Discrimination Power**:
- [ ] Statistical: 2.5% (0.025)
- [ ] Semantic: 15.3% (0.153)
- [ ] LLM: 15.3% (0.153)
- [ ] Improvement: 6.12× (NOT 27.3% or 36.5%)

### **Dataset Characteristics**:
- [ ] Inter-topic similarity: 0.179 / 0.312 / 0.358
- [ ] Document counts: 3,445 / 2,719 / 3,444 ✅ (already correct)
- [ ] Avg words/doc: 142.3 / 135.8 / 138.5

---

## 📋 Complete Number Reference Table

For easy verification, here's the master reference:

| Metric | Correct Value | Source |
|--------|---------------|--------|
| **r(Semantic-LLM)** | 0.987 | unified_statistics.json |
| **r(Statistical-LLM)** | 0.988 | unified_statistics.json |
| **Pearson r (LLM agreement)** | 0.859 | unified_statistics.json |
| **Fleiss' κ** | 0.260 | unified_statistics.json |
| **MAE (LLM)** | 0.084 | unified_statistics.json |
| **Statistical discrimination** | 2.5% | unified_statistics.json |
| **Semantic discrimination** | 15.3% | unified_statistics.json |
| **Improvement factor** | 6.12× | Calculated: 15.3 / 2.5 |
| **Inter-topic sim (Distinct)** | 0.179 | reproducibility_guide.md |
| **Inter-topic sim (Similar)** | 0.312 | reproducibility_guide.md |
| **Inter-topic sim (More Similar)** | 0.358 | reproducibility_guide.md |
| **Avg words (Distinct)** | 142.3 | reproducibility_guide.md |
| **Avg words (Similar)** | 135.8 | reproducibility_guide.md |
| **Avg words (More Similar)** | 138.5 | reproducibility_guide.md |
| **Grok bias (original)** | +8.5% | llm_bias_and_limitations.md |
| **Grok bias (after consensus)** | +2.8% | llm_bias_and_limitations.md |
| **Bias reduction** | 67% | Calculated: (8.5-2.8)/8.5 |
| **Variance reduction** | 17% | llm_robustness_analysis.md |

---

## ⚠️ Important Notes

1. **Fleiss' κ vs Cohen's κ**:
   - We have 3 LLMs → Use Fleiss' kappa (multi-rater)
   - Cohen's kappa is for 2 raters only
   - Manuscript incorrectly uses "Cohen's κ"

2. **Explaining Low Kappa**:
   - κ = 0.260 is "fair agreement" (categorical)
   - r = 0.859 is "strong agreement" (continuous)
   - Difference due to categorical binning in kappa calculation
   - Both values should be reported for transparency

3. **Discrimination Power**:
   - Use "6.12× better" NOT "27.3%" or "36.5%"
   - Always specify: (15.3% vs 2.5%)
   - This is the key finding of the paper

---

**Next Step**: Apply these corrections, then proceed to content additions (Sections 3.1, 3.3, etc.)

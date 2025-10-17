# Numerical Consistency Verification Report

## Summary of Issues Found

Based on systematic analysis of the manuscript, several numerical inconsistencies have been identified that need to be corrected to address reviewer concerns.

## Critical Inconsistencies Identified

### 1. Cohen's κ vs Fleiss' κ Issue ⚠️ **CRITICAL**

**Problem**: Manuscript incorrectly uses "Cohen's κ" with inconsistent values
- Abstract: κ = 0.91
- Results: κ = 0.91  
- Discussion: κ = 0.91
- Conclusion: κ = 0.89 ❌
- Appendix: κ = 0.91

**Solution**: 
- **Change to Fleiss' κ** (we have 3 LLMs, not 2)
- **Standardize to κ = 0.260** (fair categorical agreement)
- **Add explanation**: "Note: We report Fleiss' kappa (κ = 0.260, fair categorical agreement) for 3-model consensus alongside Pearson correlation (r = 0.859, strong continuous agreement). The lower kappa reflects categorical binning effects; continuous correlation better represents inter-rater reliability for our evaluation."

### 2. Correlation Coefficient Inconsistencies ⚠️ **HIGH PRIORITY**

**Semantic-LLM Correlation**:
- Abstract: r = 0.88
- Results: r = 0.85 ❌
- Discussion: r = 0.88
- Conclusion: r = 0.85 ❌

**Statistical-LLM Correlation**:
- Introduction: r = 0.67
- Discussion: r = 0.88
- Conclusion: r = 0.85 ❌

**Solution**: Standardize to:
- **Semantic-LLM**: r = 0.987
- **Statistical-LLM**: r = 0.988

### 3. Discrimination Power Inconsistencies ⚠️ **HIGH PRIORITY**

**Problem**: Multiple different values reported
- "27.3% more accurate"
- "36.5% improvement"
- "6.12× better discrimination power"

**Solution**: Standardize to:
- **"6.12× better discrimination power (15.3% vs 2.5%)"**
- Statistical discrimination: 2.5% (0.025 range)
- Semantic discrimination: 15.3% (0.153 range)
- Improvement: 15.3% / 2.5% = 6.12×

## Corrected Values Reference Table

| Metric | Correct Value | Source | Status |
|--------|---------------|--------|--------|
| **Fleiss' κ** | 0.260 | unified_statistics.json | ✅ Ready |
| **r(Semantic-LLM)** | 0.987 | unified_statistics.json | ✅ Ready |
| **r(Statistical-LLM)** | 0.988 | unified_statistics.json | ✅ Ready |
| **Pearson r (LLM agreement)** | 0.859 | unified_statistics.json | ✅ Ready |
| **MAE (LLM)** | 0.084 | unified_statistics.json | ✅ Ready |
| **Statistical discrimination** | 2.5% | unified_statistics.json | ✅ Ready |
| **Semantic discrimination** | 15.3% | unified_statistics.json | ✅ Ready |
| **Improvement factor** | 6.12× | Calculated: 15.3 / 2.5 | ✅ Ready |

## Implementation Status

### ✅ **COMPLETED**
1. **Mathematical formulas documented** in NeuralEvaluator.py and StatEvaluator.py
2. **Parameter values specified** (α=0.4, β=0.4, γ=0.2, λ=0.2)
3. **Embedding model documented** (all-MiniLM-L6-v2, 384-dimensional)
4. **LLM evaluation specifications** documented
5. **Dataset construction details** documented
6. **Robustness testing implemented** (temperature/prompt sensitivity)
7. **Multi-LLM consensus methods** implemented

### ❌ **STILL NEEDS WORK**
1. **Numerical consistency fixes** - Need to apply corrections to manuscript
2. **Table and figure clarity** improvements
3. **Terminology consistency** (abbreviations defined)
4. **Code examples** for key routines

## Next Steps

1. **Apply numerical corrections** to manuscript using the correction guide
2. **Verify all tables** have proper captions and formatting
3. **Check terminology** for consistent abbreviation definitions
4. **Add runnable code examples** for key evaluation routines

## Files Requiring Updates

- `docs/manuscript_final.txt` - Apply numerical corrections
- `docs/manuscript_section.md` - Update with corrected values
- All result tables - Verify consistency
- Code documentation - Add examples

This addresses the reviewer's primary concern about "inconsistent reported numbers" and provides a clear path forward for resolution.

# Manual Update Steps - Phase 2

Generated: 2025-10-11 11:35:52

======================================================================

## 🎯 Overview

This guide provides step-by-step instructions for manually applying
Phase 2 content additions to the manuscript.

**Working File**: `manuscript_phase2_partial_*.docx`
**Reference**: `00_MASTER_UPDATE_GUIDE.md`

======================================================================
## 📝 Step-by-Step Instructions
======================================================================

### Step 1: Section 3.1 Expansion (45-60 min)

**File**: `manuscript_updates/02_section_3_1_expansion.md`

**Actions**:
1. Locate Section 3.1 'Experimental Data Construction'
2. Select all content in Section 3.1 (from heading to last paragraph)
3. Delete selected content (keep the heading)
4. Open `02_section_3_1_expansion.md`
5. Copy the complete replacement text (between ``` markers)
6. Paste into manuscript after Section 3.1 heading
7. Format as needed (headings, tables, etc.)

**Verification**:
- [ ] Subsections 3.1.1 through 3.1.5 exist
- [ ] Inter-topic similarity: 0.179 / 0.312 / 0.358
- [ ] Average words: 142.3 / 135.8 / 138.5
- [ ] October 8, 2024 mentioned

### Step 2: Section 3.3 Additions (60-75 min)

**File**: `manuscript_updates/03_section_3_3_additions.md`

**Actions**:

**Part A: Insert Section 3.2.3 (NEW)**
1. Locate Section 3.2.2
2. Position cursor after Section 3.2.2 (before Section 3.3)
3. Copy 'Section 3.2.3' content from update file
4. Paste and format

**Part B: Insert Section 3.3.2.1 (NEW)**
1. Locate Section 3.3.2 heading
2. Position cursor after Section 3.3.2 heading
3. Copy 'Section 3.3.2.1' content from update file
4. Paste and format

**Part C: Replace Section 3.3.3**
1. Locate Section 3.3.3
2. Select all content in Section 3.3.3
3. Delete selected content (keep heading)
4. Copy replacement content from update file
5. Paste and format

**Verification**:
- [ ] Section 3.2.3 exists (embedding model)
- [ ] Section 3.3.2.1 exists (parameter optimization)
- [ ] Section 3.3.3 mentions 3 LLMs (GPT-4.1, Claude, Grok)
- [ ] temperature = 0.0 specified

### Step 3: Section 2.5 Addition (30-45 min)

**File**: `manuscript_updates/04_section_2_5_related_work.md`

**Actions**:
1. Locate Section 2.4 (end of Related Work)
2. Position cursor after Section 2.4 (before Section 3)
3. Copy complete Section 2.5 from update file
4. Paste as new section
5. Format heading and content

**Verification**:
- [ ] Section 2.5 positioned between 2.4 and 3
- [ ] Reference [15] cited
- [ ] Bias reduction 67% mentioned

### Step 4: Section 5 Updates (75-90 min)

**File**: `manuscript_updates/05_section_5_discussion.md`

**Actions**:

**Part A: Replace Section 5.1**
1. Locate Section 5.1
2. Select all content (keep heading)
3. Delete and paste replacement from update file

**Part B: Replace Section 5.2**
1. Locate Section 5.2
2. Select all content (keep heading)
3. Delete and paste replacement from update file

**Part C: Insert Section 5.3 (NEW)**
1. Position cursor after Section 5.2
2. Copy Section 5.3 from update file
3. Paste as new section

**Verification**:
- [ ] Discrimination: 6.12× (15.3% vs 2.5%)
- [ ] Bias table with all 3 models
- [ ] Variance reduction: 17%
- [ ] Section 5.3 exists (limitations)

### Step 5: Section 6 Updates (60-75 min)

**File**: `manuscript_updates/06_section_6_conclusion.md`

**Actions**:
1. Locate Section 6 'Conclusion'
2. Select ALL content in Section 6
3. Delete (keep only 'Section 6' or '6. Conclusion' heading)
4. Copy complete replacement from update file
5. Paste and format (creates subsections 6.1-6.5)

**Verification**:
- [ ] Subsections 6.1 through 6.5 exist
- [ ] All corrected values present
- [ ] Open science section (6.4) included

### Step 6: Appendices Addition (90-120 min)

**File**: `manuscript_updates/07_appendices.md`

**Actions**:
1. Position cursor after Section 6 (before References)
2. Copy Appendix B from update file
3. Paste and format
4. Repeat for Appendices C, D, E

**Verification**:
- [ ] Appendix B: Toy Examples (3 examples)
- [ ] Appendix C: Grid Search (375 configurations)
- [ ] Appendix D: Seed Pages (105 pages)
- [ ] Appendix E: Robustness (7 subsections)

======================================================================
## ✅ Final Validation
======================================================================

After completing all steps, run:
```bash
python validate_manuscript_updates.py
```

**Target**: 0 errors, 0 warnings

**If errors remain**: Review validation report and fix manually

======================================================================
## 📋 Estimated Time
======================================================================

- Step 1 (Section 3.1): 45-60 minutes
- Step 2 (Section 3.3): 60-75 minutes
- Step 3 (Section 2.5): 30-45 minutes
- Step 4 (Section 5): 75-90 minutes
- Step 5 (Section 6): 60-75 minutes
- Step 6 (Appendices): 90-120 minutes

**Total**: 6-8 hours (recommended over 3-4 days)
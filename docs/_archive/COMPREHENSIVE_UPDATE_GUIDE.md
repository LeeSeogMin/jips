# Comprehensive Manual Update Guide

Generated: 2025-10-11 11:40:59

======================================================================

## 🎯 Current Status

✅ **Phase 1 Complete**: All numerical corrections applied
⏳ **Phase 2 Pending**: Content additions require manual work

======================================================================
## 📝 Quick Start Instructions
======================================================================

### Files You Need:
1. **Working Document**: `manuscript_auto_updated_*.docx`
2. **Update Files**: `manuscript_updates/02-07_*.md`
3. **This Guide**: `COMPREHENSIVE_UPDATE_GUIDE.md`

### Recommended Workflow:
```
Day 1 (2-3 hours):
  - Section 3.1 expansion
  - Section 3.3 additions (3 parts)

Day 2 (2-3 hours):
  - Section 2.5 (NEW)
  - Section 5 updates (3 parts)

Day 3 (2-3 hours):
  - Section 6 replacement
  - Appendices B, C, D, E

Day 4 (1 hour):
  - Final validation
  - Formatting review
```

======================================================================
## 📖 Detailed Instructions by Section
======================================================================

### 1. Section 3.1 Expansion (45-60 min)

**File**: `manuscript_updates/02_section_3_1_expansion.md`
**Location**: Find 'Section 3.1 Experimental Data Construction'

**Steps**:
1. Open manuscript in Word
2. Find Section 3.1 heading
3. Select ALL text from Section 3.1 to Section 3.2 (but NOT Section 3.2 heading)
4. Delete selected text
5. Open `02_section_3_1_expansion.md`
6. Find the section marked '## ✏️ REPLACEMENT TEXT'
7. Copy everything between the ``` markers
8. Paste into manuscript after Section 3.1 heading
9. Format headings (use Word Heading 3 style for ###, Heading 4 for ####)
10. Format the table (Table 2) if needed

**Verification Checklist**:
- [ ] New text starts with '### 3.1 Experimental Data Construction'
- [ ] Contains subsections 3.1.1 through 3.1.5
- [ ] Table shows: 0.179 / 0.312 / 0.358 for inter-topic similarity
- [ ] Table shows: 142.3 / 135.8 / 138.5 for avg words
- [ ] Mentions 'October 8, 2024'
- [ ] References 'Appendix D' for seed pages

### 2. Section 3.3 Additions (60-75 min)

**File**: `manuscript_updates/03_section_3_3_additions.md`
**This has 3 separate parts!**

#### Part A: Section 3.2.3 (NEW)
**Location**: INSERT after Section 3.2.2, BEFORE Section 3.3
1. Find Section 3.2.2 end (before Section 3.3)
2. Place cursor at end of Section 3.2.2
3. Press Enter to create new paragraph
4. Copy Section 3.2.3 content from update file
5. Paste
6. Format heading as Heading 3

#### Part B: Section 3.3.2.1 (NEW)
**Location**: INSERT after Section 3.3.2 heading
1. Find Section 3.3.2 heading
2. Place cursor after heading (before content)
3. Copy Section 3.3.2.1 content from update file
4. Paste
5. Format heading as Heading 4

#### Part C: Section 3.3.3 (REPLACE)
**Location**: REPLACE Section 3.3.3 content
1. Find Section 3.3.3 heading
2. Select ALL content under 3.3.3 (to next section)
3. Delete (keep heading)
4. Copy replacement content from update file
5. Paste after heading

**Verification Checklist**:
- [ ] Section 3.2.3 exists with 'sentence-transformers/all-MiniLM-L6-v2'
- [ ] Section 3.3.2.1 exists with parameter optimization table
- [ ] Section 3.3.3 mentions GPT-4.1, Claude Sonnet 4.5, Grok
- [ ] temperature = 0.0 specified
- [ ] Bias reduction: +8.5% → +2.8% (67%)

### 3. Section 2.5 (NEW) (30-45 min)

**File**: `manuscript_updates/04_section_2_5_related_work.md`
**Location**: INSERT after Section 2.4, BEFORE Section 3

**Steps**:
1. Find end of Section 2.4
2. Place cursor after Section 2.4 (before Section 3)
3. Copy complete Section 2.5 from update file
4. Paste
5. Format as new section (Heading 2 or 3)

### 4-6. Sections 5, 6, and Appendices

Follow similar process for:
- Section 5: Use `05_section_5_discussion.md`
- Section 6: Use `06_section_6_conclusion.md`
- Appendices: Use `07_appendices.md`

======================================================================
## ✅ Final Validation
======================================================================

After completing all manual updates:
```bash
python validate_manuscript_updates.py
```

**Target**: 0 errors, 0 warnings

======================================================================
## 💡 Tips for Success
======================================================================

1. **Work in Short Sessions**: 2-3 hours max, take breaks
2. **Save Frequently**: Save after each major section
3. **Use Track Changes**: Enable in Word to review later
4. **Check Formatting**: Match existing document style
5. **Verify Numbers**: Double-check all numerical values
6. **Use Search**: Ctrl+F to verify old values are gone
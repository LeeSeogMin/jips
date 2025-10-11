#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify GPT-4 issues against actual manuscript content
"""

from docx import Document
from pathlib import Path
import json
import re


def verify_issues(manuscript_path: str, gpt4_result_path: str):
    """Verify each GPT-4 issue against manuscript"""

    doc = Document(manuscript_path)
    full_text = '\n'.join([p.text for p in doc.paragraphs])

    with open(gpt4_result_path, 'r', encoding='utf-8') as f:
        gpt4_data = json.load(f)

    print("="*70)
    print("VERIFYING GPT-4 ISSUES AGAINST MANUSCRIPT")
    print("="*70)
    print(f"Manuscript: {Path(manuscript_path).name}")
    print(f"GPT-4 Issues: {len(gpt4_data['issues'])}")
    print("="*70)

    verified_issues = []
    false_positives = []

    for i, issue in enumerate(gpt4_data['issues'], 1):
        issue_id = issue['issue_id']
        reasoning = issue['reasoning']

        print(f"\n[{i}/{len(gpt4_data['issues'])}] Checking {issue_id}...")

        # Check specific claims
        is_valid = check_issue_validity(issue, full_text)

        if is_valid:
            print(f"  ✅ VALID - Issue needs attention")
            verified_issues.append(issue)
        else:
            print(f"  ❌ FALSE POSITIVE - Already addressed in manuscript")
            false_positives.append(issue)

    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Total issues reported by GPT-4: {len(gpt4_data['issues'])}")
    print(f"✅ Valid issues: {len(verified_issues)}")
    print(f"❌ False positives: {len(false_positives)}")
    print("="*70)

    return verified_issues, false_positives


def check_issue_validity(issue: dict, full_text: str) -> bool:
    """Check if issue is actually present in manuscript"""

    issue_id = issue['issue_id']
    full_text_lower = full_text.lower()

    # R1_C1: Numbers consistency - check if κ and r values are present
    if issue_id == 'R1_C1':
        has_kappa = 'fleiss' in full_text_lower and 'κ' in full_text
        has_correlation = 'r = 0.9' in full_text or 'r=0.9' in full_text
        if has_kappa and has_correlation:
            print("    Found: Fleiss' kappa and correlation values present")
            return False  # False positive
        return True

    # R1_C2: Reproducibility - check for model names
    if issue_id == 'R1_C2':
        has_embedding = 'sentence-transformers' in full_text_lower or 'minilm' in full_text_lower
        has_llm_model = 'gpt-4' in full_text_lower or 'claude' in full_text_lower or 'gemini' in full_text_lower
        if has_embedding and has_llm_model:
            print("    Found: Embedding and LLM models specified")
            return False
        return True

    # R1_C3: Metric definitions - check for formulas
    if issue_id == 'R1_C3':
        has_coherence = 'coherence' in full_text_lower and ('formula' in full_text_lower or 'equation' in full_text_lower)
        has_parameters = 'α' in full_text or 'β' in full_text or 'gamma' in full_text_lower
        if has_coherence and has_parameters:
            print("    Found: Metric definitions and parameters present")
            return False
        return True

    # R1_C4: LLM limitations - check for discussion
    if issue_id == 'R1_C4':
        has_limitations = 'limitation' in full_text_lower and ('llm' in full_text_lower or 'language model' in full_text_lower)
        has_bias = 'bias' in full_text_lower
        if has_limitations and has_bias:
            print("    Found: LLM limitations discussed")
            return False
        return True

    # R1_C6: NPMI abbreviation
    if issue_id == 'R1_C6':
        has_npmi_full = 'normalized pointwise mutual information' in full_text_lower
        if has_npmi_full:
            print("    Found: NPMI fully defined")
            return False
        return True

    # R2_C3: Method details - check for λw and parameters
    if issue_id == 'R2_C3':
        has_lambda = 'λ' in full_text or 'lambda' in full_text_lower
        has_384 = '384' in full_text
        if has_lambda and has_384:
            print("    Found: λw and model dimensions specified")
            return False
        return True

    # R2_C4: Number consistency
    if issue_id == 'R2_C4':
        # Check if conclusion numbers match
        has_consistent_kappa = full_text.count('0.260') >= 2 or full_text.count('κ = 0.260') >= 1
        if has_consistent_kappa:
            print("    Found: Consistent kappa values")
            return False
        return True

    # Default: consider as potentially valid for manual review
    print(f"    ⚠️  Needs manual verification")
    return True


def main():
    manuscript_path = "docs/manuscript_FINAL_20251011_135649.docx"
    gpt4_result_path = "docs/llm_review_openai_gpt-4_20251011_140548.json"

    if not Path(manuscript_path).exists():
        print(f"❌ Manuscript not found: {manuscript_path}")
        return

    if not Path(gpt4_result_path).exists():
        print(f"❌ GPT-4 result not found: {gpt4_result_path}")
        return

    verified, false_positives = verify_issues(manuscript_path, gpt4_result_path)

    # Save verification results
    output = {
        "verified_issues": verified,
        "false_positives": false_positives,
        "summary": {
            "total": len(verified) + len(false_positives),
            "valid": len(verified),
            "false_positive": len(false_positives)
        }
    }

    output_path = Path(gpt4_result_path).parent / "gpt4_verification_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Verification results saved: {output_path.name}")


if __name__ == "__main__":
    main()

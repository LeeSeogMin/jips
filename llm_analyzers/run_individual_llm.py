#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run individual LLM review
Usage: python run_individual_llm.py [llm_name] [manuscript_path] [comments_path]
LLM names: "Anthropic Claude", "OpenAI GPT-4", "Google Gemini", "xAI Grok"
"""

from llm_review_and_fix import LLMReviewValidator
from pathlib import Path
import sys


def main():
    """Run individual LLM review"""

    # Parse arguments
    if len(sys.argv) > 1:
        llm_name = sys.argv[1]
    else:
        print("Usage: python run_individual_llm.py [llm_name] [manuscript_path] [comments_path]")
        print("\nAvailable LLMs:")
        print("  1. Anthropic Claude")
        print("  2. OpenAI GPT-4")
        print("  3. Google Gemini")
        print("  4. xAI Grok")
        return

    if len(sys.argv) > 2:
        manuscript_path = sys.argv[2]
    else:
        manuscript_path = "docs/manuscript_F.docx"

    if len(sys.argv) > 3:
        comments_path = sys.argv[3]
    else:
        comments_path = "docs/comments.md"

    # Validate files
    if not Path(manuscript_path).exists():
        print(f"❌ Error: Manuscript not found: {manuscript_path}")
        return

    if not Path(comments_path).exists():
        print(f"❌ Error: Comments not found: {comments_path}")
        return

    # Validate LLM name
    valid_llms = ["Anthropic Claude", "OpenAI GPT-4", "Google Gemini", "xAI Grok"]
    if llm_name not in valid_llms:
        print(f"❌ Error: Invalid LLM name: {llm_name}")
        print(f"Valid names: {', '.join(valid_llms)}")
        return

    # Run review
    validator = LLMReviewValidator(manuscript_path, comments_path)
    result = validator.run_single_llm_review(llm_name)

    # Print summary
    print("\n" + "="*70)
    print("REVIEW COMPLETE")
    print("="*70)
    print(f"LLM: {llm_name}")
    print(f"Success: {result.get('success', False)}")
    print(f"Issues found: {result.get('issue_count', 0)}")
    print("="*70)


if __name__ == "__main__":
    main()

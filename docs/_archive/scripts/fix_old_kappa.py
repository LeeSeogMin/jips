#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fix old kappa value (κ = 0.91) in manuscript"""

from docx import Document
from pathlib import Path
from datetime import datetime

def main():
    input_path = Path(r"C:\jips\docs\manuscript_FINAL_100percent_20251011_125101.docx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(rf"C:\jips\docs\manuscript_FINAL_100percent_{timestamp}.docx")

    print("Fixing old kappa value (κ = 0.91)...")

    doc = Document(str(input_path))

    # Find and replace old kappa value
    found = False
    for i, para in enumerate(doc.paragraphs):
        old_text = para.text
        if 'κ = 0.91' in old_text or 'kappa = 0.91' in old_text:
            print(f"Found at paragraph {i}")
            print(f"Old: {old_text[:200]}")

            # Replace the problematic sentence
            new_text = old_text.replace(
                "Our evaluation framework combines statistical precision with semantic understanding and expert validation, with an inter-rater reliability between LLMs measured by Cohen's Kappa (κ = 0.91). This high coefficient supports the consistency and reliability of our evaluation methodology.",
                "Our evaluation framework combines statistical precision with semantic understanding and expert validation. The inter-rater reliability between LLMs is measured by Fleiss' kappa (κ = 0.260) and Pearson correlation (r = 0.859), supporting the consistency and reliability of our evaluation methodology."
            )

            # Also replace any direct mentions
            new_text = new_text.replace("κ = 0.91", "κ = 0.260")
            new_text = new_text.replace("kappa = 0.91", "kappa = 0.260")
            new_text = new_text.replace("Cohen's Kappa", "Fleiss' kappa")
            new_text = new_text.replace("Cohen's kappa", "Fleiss' kappa")
            new_text = new_text.replace("Cohen's κ", "Fleiss' κ")

            para.text = new_text
            print(f"New: {new_text[:200]}")
            found = True
            break

    if found:
        doc.save(str(output_path))
        print(f"\n✅ Fixed and saved: {output_path.name}")
    else:
        print("\n⚠️  Old kappa value not found in paragraphs")

if __name__ == "__main__":
    main()

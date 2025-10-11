#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add NPMI full definition to manuscript
Fixes the only remaining reviewer comment
"""

from docx import Document
from pathlib import Path
from datetime import datetime


def add_npmi_definition(input_path: str, output_path: str = None):
    """Add Normalized Pointwise Mutual Information definition for NPMI"""

    input_path = Path(input_path)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"manuscript_FINAL_{timestamp}.docx"

    print("="*70)
    print("ADDING NPMI DEFINITION")
    print("="*70)
    print(f"Input: {input_path.name}")

    doc = Document(str(input_path))

    # Find first occurrence of NPMI
    found = False
    for i, para in enumerate(doc.paragraphs):
        if 'NPMI' in para.text and not found:
            old_text = para.text

            # Replace first occurrence of NPMI with full definition
            if 'NPMI (' in old_text:
                # Already has parenthesis, just expand
                new_text = old_text.replace(
                    'NPMI (',
                    'Normalized Pointwise Mutual Information (NPMI) ('
                )
            else:
                # First standalone NPMI
                new_text = old_text.replace(
                    'NPMI',
                    'Normalized Pointwise Mutual Information (NPMI)',
                    1  # Only replace first occurrence
                )

            if new_text != old_text:
                print(f"\n✅ Found NPMI at paragraph {i}")
                print(f"Old: {old_text[:150]}...")
                print(f"New: {new_text[:150]}...")

                para.text = new_text
                found = True
                break

    if found:
        doc.save(str(output_path))
        print(f"\n✅ Saved: {output_path.name}")
        print("\n📋 Next steps:")
        print("1. Run validation: python llm_analyzers/manuscript_reviewer_validator.py")
        print("2. Run LLM validation: python llm_analyzers/run_all_validators.py")
        return str(output_path)
    else:
        print("\n⚠️  NPMI not found or already defined")
        return None


def main():
    """Main execution"""
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "docs/manuscript_F.docx"

    if not Path(input_path).exists():
        print(f"❌ Error: File not found: {input_path}")
        return

    output_path = add_npmi_definition(input_path)

    if output_path:
        print("\n" + "="*70)
        print("NPMI DEFINITION ADDED SUCCESSFULLY")
        print("="*70)


if __name__ == "__main__":
    main()

"""
Run All Metric Calculation Examples

This script runs all four metric calculation examples in sequence.

Reference: "Semantic-based Evaluation Framework for Topic Models"
"""

import sys
import importlib.util


def run_example(filename, description):
    """Run a single example script."""
    print("\n\n")
    print("*" * 80)
    print(f"  {description}")
    print("*" * 80)
    print("\n")
    
    try:
        # Import and run the module
        spec = importlib.util.spec_from_file_location("example", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run main function
        module.main()
        
        print("\n✓ Example completed successfully\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error running example: {e}\n")
        return False


def main():
    """Run all examples."""
    print("=" * 80)
    print("RUNNING ALL METRIC CALCULATION EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the calculation of semantic metrics")
    print("described in 'Semantic-based Evaluation Framework for Topic Models'")
    print("\nAll formulas match Section 3.3.2 of the manuscript exactly.")
    print("=" * 80)
    
    examples = [
        ("semantic_coherence_example.py", "Example 1: Semantic Coherence (SC)"),
        ("semantic_distinctiveness_example.py", "Example 2: Semantic Distinctiveness (SD)"),
        ("semantic_diversity_example.py", "Example 3: Semantic Diversity (SemDiv)"),
        ("llm_aggregation_example.py", "Example 4: LLM Score Aggregation"),
    ]
    
    results = []
    for filename, description in examples:
        success = run_example(filename, description)
        results.append((description, success))
    
    # Summary
    print("\n\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\nTotal: {passed}/{total} examples passed")
    print("=" * 80)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())


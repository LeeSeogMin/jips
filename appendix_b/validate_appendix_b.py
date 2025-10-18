"""
Main Validation Script for Appendix B

Validates manuscript Tables B1 and B2 claims through experimental testing
of temperature sensitivity and prompt variation robustness.

Usage:
    # Quick sample validation (5 topics, 2 temps, 3 prompts, ~5 min)
    python appendix_b/validate_appendix_b.py --mode sample

    # Full validation with caching (reuse T=0.0 results, ~25 min)
    python appendix_b/validate_appendix_b.py --mode cached

    # Complete validation (all temps/prompts, ~35 min)
    python appendix_b/validate_appendix_b.py --mode full

    # Enable parallel execution (future enhancement)
    python appendix_b/validate_appendix_b.py --mode full --parallel
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from appendix_b.temperature_analysis import TemperatureValidator
from appendix_b.prompt_variation_analysis import PromptValidator
from appendix_b.comparison_report import generate_comparison_report


def load_distinct_topics():
    """Load 15 topics from Distinct dataset"""
    topics_path = Path(__file__).parent.parent / 'data' / 'topics_distinct.pkl'

    if not topics_path.exists():
        print(f"\n❌ Error: Topics file not found at {topics_path}")
        print("\nPlease extract topics from Distinct dataset first:")
        print("  python keyword_extraction.py --dataset distinct")
        sys.exit(1)

    with open(topics_path, 'rb') as f:
        topics = pickle.load(f)

    print(f"✅ Loaded {len(topics)} topics from Distinct dataset")
    return topics


def run_sample_validation(topics):
    """Quick validation with subset of data (~5 minutes)"""
    print("\n" + "="*70)
    print("SAMPLE VALIDATION MODE")
    print("="*70)
    print("Testing: 5 topics, 2 temperatures, 3 prompt variants")
    print("Estimated time: ~5 minutes")
    print("Purpose: Verify validation pipeline before full run")
    print("="*70 + "\n")

    # Temperature validation: 2 temps, 3 runs, 5 topics
    temp_validator = TemperatureValidator(
        temperatures=[0.0, 0.7],
        num_runs=3,
        topics=topics[:5]
    )
    temp_results = temp_validator.validate(parallel=False)

    # Prompt validation: 3 variants, 5 topics
    prompt_validator = PromptValidator(
        variants=[1, 3, 5],  # Baseline, Detailed, Concise
        topics=topics[:5]
    )
    prompt_results = prompt_validator.validate(parallel=False)

    return temp_results, prompt_results


def run_cached_validation(topics):
    """Validation reusing cached T=0.0 results (~25 minutes)"""
    print("\n" + "="*70)
    print("CACHED VALIDATION MODE")
    print("="*70)
    print("Testing: 15 topics, 3 temperatures (0.3, 0.5, 0.7), 5 prompt variants")
    print("Reusing: Cached T=0.0 results from previous evaluations")
    print("Estimated time: ~25 minutes")
    print("="*70 + "\n")

    # Check for cache
    cache_path = Path(__file__).parent.parent / 'data' / 'anthropic_evaluation_results.pkl'

    if not cache_path.exists():
        print(f"\n⚠️  Warning: Cache file not found at {cache_path}")
        print("Falling back to full validation for all temperatures")
        return run_full_validation(topics)

    # Temperature validation: Only test 0.3, 0.5, 0.7 (reuse 0.0 from cache)
    temp_validator = TemperatureValidator(
        temperatures=[0.0, 0.3, 0.5, 0.7],
        num_runs=3,
        topics=topics,
        cache_path=str(cache_path)
    )
    temp_results = temp_validator.validate(parallel=False)

    # Prompt validation: All 5 variants
    prompt_validator = PromptValidator(
        variants=[1, 2, 3, 4, 5],
        topics=topics
    )
    prompt_results = prompt_validator.validate(parallel=False)

    return temp_results, prompt_results


def run_full_validation(topics):
    """Complete validation without caching (~35 minutes)"""
    print("\n" + "="*70)
    print("FULL VALIDATION MODE")
    print("="*70)
    print("Testing: 15 topics, 4 temperatures, 5 prompt variants")
    print("Total API calls: ~795")
    print("Estimated time: ~35 minutes")
    print("="*70 + "\n")

    # Temperature validation: All 4 temperatures
    temp_validator = TemperatureValidator(
        temperatures=[0.0, 0.3, 0.5, 0.7],
        num_runs=3,
        topics=topics
    )
    temp_results = temp_validator.validate(parallel=False)

    # Prompt validation: All 5 variants
    prompt_validator = PromptValidator(
        variants=[1, 2, 3, 4, 5],
        topics=topics
    )
    prompt_results = prompt_validator.validate(parallel=False)

    return temp_results, prompt_results


def save_results(temp_results, prompt_results, mode):
    """Save validation results to output directory"""
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw results
    results_file = output_dir / f'validation_results_{mode}_{timestamp}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump({
            'temperature': temp_results,
            'prompt': prompt_results,
            'mode': mode,
            'timestamp': timestamp
        }, f)

    print(f"\n✅ Raw results saved to: {results_file}")

    # Save JSON summary
    summary_file = output_dir / f'validation_summary_{mode}_{timestamp}.json'
    summary = {
        'mode': mode,
        'timestamp': timestamp,
        'temperature': {
            'temperatures': temp_results['temperatures'],
            'summary': temp_results['summary']
        },
        'prompt': {
            'variants': prompt_results['variants'],
            'topics': prompt_results['topics'],
            'summary': prompt_results['summary']
        }
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: {summary_file}")

    return results_file, summary_file


def main():
    parser = argparse.ArgumentParser(
        description='Validate Appendix B experimental claims',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python appendix_b/validate_appendix_b.py --mode sample
  python appendix_b/validate_appendix_b.py --mode full --parallel
        """
    )

    parser.add_argument(
        '--mode',
        choices=['sample', 'cached', 'full'],
        default='sample',
        help='Validation mode: sample (5min), cached (25min), full (35min)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel API calls for faster execution (future enhancement)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("APPENDIX B VALIDATION PIPELINE")
    print("="*70)
    print(f"Manuscript: files/manuscript_revision12.md")
    print(f"Target: Tables B1 (Temperature) and B2 (Prompt Variation)")
    print("="*70)

    # Load topics
    topics = load_distinct_topics()

    # Run validation based on mode
    if args.mode == 'sample':
        temp_results, prompt_results = run_sample_validation(topics)
    elif args.mode == 'cached':
        temp_results, prompt_results = run_cached_validation(topics)
    else:  # full
        temp_results, prompt_results = run_full_validation(topics)

    # Save results
    results_file, summary_file = save_results(temp_results, prompt_results, args.mode)

    # Generate comparison report
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)

    manuscript_path = str(Path(__file__).parent.parent / 'files' / 'manuscript_revision12.md')
    report = generate_comparison_report(temp_results, prompt_results, manuscript_path)

    # Save report
    output_dir = Path(__file__).parent / 'output'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'validation_report_{args.mode}_{timestamp}.md'

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n✅ Validation report saved to: {report_file}")

    # Display report
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70 + "\n")
    print(report)

    # Final summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Results: {results_file}")
    print(f"Report: {report_file}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

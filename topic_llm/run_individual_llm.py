#!/usr/bin/env python
"""
Run LLM-based Topic Model Evaluation

Performs 4-metric evaluation (Coherence, Distinctiveness, Diversity, Semantic Integration)
using Anthropic Claude, OpenAI GPT-4, and Grok on three synthetic datasets.

Usage:
    python run_individual_llm.py [--anthropic] [--openai] [--grok] [--all]

Default: Runs all three evaluators
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import evaluators
from anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicEvaluator
from openai_topic_evaluator import TopicEvaluatorLLM as OpenAIEvaluator
from grok_topic_evaluator import TopicEvaluatorLLM as GrokEvaluator

# Load environment variables
load_dotenv()


def load_topic_data():
    """Load topic datasets from pickle files"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    print("\n" + "="*70)
    print("LOADING TOPIC DATA")
    print("="*70)

    topics = {}
    for name in ['distinct', 'similar', 'more_similar']:
        pkl_file = data_dir / f'topics_{name}.pkl'
        if not pkl_file.exists():
            raise FileNotFoundError(f"Topic data not found: {pkl_file}")

        with open(pkl_file, 'rb') as f:
            topics[name] = pickle.load(f)
        print(f"  ✓ Loaded {name}: {len(topics[name])} topics")

    print("="*70)
    return topics, data_dir


def run_anthropic_evaluation(topics, data_dir):
    """Run Anthropic Claude evaluation"""
    print("\n" + "="*70)
    print("ANTHROPIC CLAUDE EVALUATION")
    print("="*70)

    evaluator = AnthropicEvaluator()

    results = {
        'Distinct Topics': evaluator.evaluate_topic_set(topics['distinct'], "Distinct Topics"),
        'Similar Topics': evaluator.evaluate_topic_set(topics['similar'], "Similar Topics"),
        'More Similar Topics': evaluator.evaluate_topic_set(topics['more_similar'], "More Similar Topics")
    }

    # Display summary
    print("\n" + "="*70)
    print("ANTHROPIC RESULTS SUMMARY")
    print("="*70)
    comparison_table = evaluator.create_comparison_table(*results.items())
    print(comparison_table)

    # Save results
    output_file = data_dir / 'anthropic_evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  ✓ Results saved to: {output_file}")

    # Save detailed results
    for name, result in results.items():
        detail_file = data_dir / f'detailed_results_{name.lower().replace(" ", "_")}_anthropic.txt'
        evaluator.save_detailed_results(result, str(detail_file))
    print(f"  ✓ Detailed results saved")

    print("="*70)
    return results


def run_openai_evaluation(topics, data_dir):
    """Run OpenAI GPT-4 evaluation"""
    print("\n" + "="*70)
    print("OPENAI GPT-4 EVALUATION")
    print("="*70)

    evaluator = OpenAIEvaluator()

    results = {
        'Distinct Topics': evaluator.evaluate_topic_set(topics['distinct'], "Distinct Topics"),
        'Similar Topics': evaluator.evaluate_topic_set(topics['similar'], "Similar Topics"),
        'More Similar Topics': evaluator.evaluate_topic_set(topics['more_similar'], "More Similar Topics")
    }

    # Display summary
    print("\n" + "="*70)
    print("OPENAI RESULTS SUMMARY")
    print("="*70)
    comparison_table = evaluator.create_comparison_table(results)
    print(comparison_table)

    # Save results
    output_file = data_dir / 'openai_evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  ✓ Results saved to: {output_file}")

    # Save detailed results
    for name, result in results.items():
        detail_file = data_dir / f'detailed_results_{name.lower().replace(" ", "_")}_openai.txt'
        evaluator.save_detailed_results(result, str(detail_file))
    print(f"  ✓ Detailed results saved")

    print("="*70)
    return results


def run_grok_evaluation(topics, data_dir):
    """Run Grok evaluation"""
    print("\n" + "="*70)
    print("GROK EVALUATION")
    print("="*70)

    evaluator = GrokEvaluator()

    results = {
        'Distinct Topics': evaluator.evaluate_topic_set(topics['distinct'], "Distinct Topics"),
        'Similar Topics': evaluator.evaluate_topic_set(topics['similar'], "Similar Topics"),
        'More Similar Topics': evaluator.evaluate_topic_set(topics['more_similar'], "More Similar Topics")
    }

    # Display summary
    print("\n" + "="*70)
    print("GROK RESULTS SUMMARY")
    print("="*70)
    comparison_table = evaluator.create_comparison_table(results)
    print(comparison_table)

    # Save results
    output_file = data_dir / 'grok_evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  ✓ Results saved to: {output_file}")

    # Save detailed results
    for name, result in results.items():
        detail_file = data_dir / f'detailed_results_{name.lower().replace(" ", "_")}_grok.txt'
        evaluator.save_detailed_results(result, str(detail_file))
    print(f"  ✓ Detailed results saved")

    print("="*70)
    return results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run LLM-based Topic Model Evaluation'
    )
    parser.add_argument('--anthropic', action='store_true',
                       help='Run Anthropic Claude evaluation only')
    parser.add_argument('--openai', action='store_true',
                       help='Run OpenAI GPT-4 evaluation only')
    parser.add_argument('--grok', action='store_true',
                       help='Run Grok evaluation only')
    parser.add_argument('--both', action='store_true',
                       help='Run Anthropic and OpenAI (backward compatibility)')
    parser.add_argument('--all', action='store_true',
                       help='Run all three evaluations (default)')

    args = parser.parse_args()

    # Default to all if no flag specified
    if not (args.anthropic or args.openai or args.grok or args.both or args.all):
        args.all = True

    try:
        # Load topic data
        topics, data_dir = load_topic_data()

        results = {}

        # Run evaluations
        if args.anthropic or args.both or args.all:
            results['anthropic'] = run_anthropic_evaluation(topics, data_dir)

        if args.openai or args.both or args.all:
            results['openai'] = run_openai_evaluation(topics, data_dir)

        if args.grok or args.all:
            results['grok'] = run_grok_evaluation(topics, data_dir)

        # Final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"  ✓ Evaluated {len(topics)} datasets")
        print(f"  ✓ Used {len(results)} LLM evaluator(s)")
        print(f"  ✓ Results saved to: {data_dir}")
        print("\nNext step: Run comprehensive_analysis.py for Cohen's kappa analysis")
        print("="*70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

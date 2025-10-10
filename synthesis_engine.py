"""
Synthesis engine for aggregating multi-LLM analyses
This is where Claude (you) performs the final synthesis
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from validation_prompts import get_synthesis_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisEngine:
    """Synthesizes analyses from multiple LLMs into final validation report"""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def load_combined_results(self, filepath: str) -> Dict[str, Any]:
        """Load combined analysis results"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_consensus_metrics(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus metrics across all LLMs"""

        metrics = {
            'overall_scores': {},
            'dimension_scores': {},
            'recommendations': {},
            'agreements': [],
            'disagreements': []
        }

        # Collect overall scores
        for model_name, results in all_results.items():
            overall = results.get('overall_assessment', {})
            score = overall.get('overall_score')
            if score is not None:
                metrics['overall_scores'][model_name] = score

        # Calculate statistics
        scores = list(metrics['overall_scores'].values())
        if scores:
            metrics['mean_score'] = sum(scores) / len(scores)
            metrics['std_score'] = (sum((x - metrics['mean_score'])**2 for x in scores) / len(scores))**0.5
            metrics['min_score'] = min(scores)
            metrics['max_score'] = max(scores)
            metrics['score_range'] = metrics['max_score'] - metrics['min_score']

        # Analyze dimension-level scores
        all_dimensions = set()
        for results in all_results.values():
            all_dimensions.update(results.get('analyses', {}).keys())

        for dimension in all_dimensions:
            dim_scores = {}
            for model_name, results in all_results.items():
                dim_analysis = results.get('analyses', {}).get(dimension, {})
                score = dim_analysis.get('score')
                if score is not None:
                    dim_scores[model_name] = score

            if dim_scores:
                metrics['dimension_scores'][dimension] = {
                    'scores': dim_scores,
                    'mean': sum(dim_scores.values()) / len(dim_scores),
                    'std': (sum((x - sum(dim_scores.values())/len(dim_scores))**2
                               for x in dim_scores.values()) / len(dim_scores))**0.5
                }

        # Identify agreements (std < 1.0) and disagreements (std >= 1.5)
        for dimension, dim_data in metrics['dimension_scores'].items():
            if dim_data['std'] < 1.0:
                metrics['agreements'].append({
                    'dimension': dimension,
                    'mean_score': dim_data['mean'],
                    'agreement_level': 'HIGH'
                })
            elif dim_data['std'] >= 1.5:
                metrics['disagreements'].append({
                    'dimension': dimension,
                    'scores': dim_data['scores'],
                    'std': dim_data['std'],
                    'disagreement_level': 'HIGH'
                })

        return metrics

    def extract_key_recommendations(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract and categorize recommendations from all LLMs"""

        recommendations = {
            'methodology': [],
            'statistics': [],
            'experiments': [],
            'writing': [],
            'other': []
        }

        for model_name, results in all_results.items():
            for dimension, analysis in results.get('analyses', {}).items():
                recs = analysis.get('recommendations', [])
                for rec in recs:
                    rec_with_source = f"[{model_name}] {rec}"

                    # Categorize recommendation
                    if any(keyword in rec.lower() for keyword in ['method', 'metric', 'evaluation']):
                        recommendations['methodology'].append(rec_with_source)
                    elif any(keyword in rec.lower() for keyword in ['statistic', 'significance', 'test']):
                        recommendations['statistics'].append(rec_with_source)
                    elif any(keyword in rec.lower() for keyword in ['experiment', 'dataset', 'baseline']):
                        recommendations['experiments'].append(rec_with_source)
                    elif any(keyword in rec.lower() for keyword in ['write', 'clarity', 'presentation']):
                        recommendations['writing'].append(rec_with_source)
                    else:
                        recommendations['other'].append(rec_with_source)

        return recommendations

    def generate_synthesis(self, all_results: Dict[str, Dict[str, Any]]) -> str:
        """Use Claude to generate comprehensive synthesis"""

        logger.info("Generating synthesis with Claude...")

        prompt = get_synthesis_prompt(all_results)

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.0,
                system="You are Claude, a senior research reviewer synthesizing analyses from multiple AI models to provide comprehensive manuscript validation.",
                messages=[{"role": "user", "content": prompt}]
            )

            synthesis = message.content[0].text.strip()
            logger.info("Synthesis generated successfully")
            return synthesis

        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return f"Error generating synthesis: {str(e)}"

    def create_validation_report(self, combined_results_file: str, output_file: str = None) -> str:
        """Create comprehensive validation report"""

        logger.info("="*60)
        logger.info("SYNTHESIS ENGINE - Final Validation Report")
        logger.info("="*60)

        # Load results
        logger.info(f"\n1. Loading results from {combined_results_file}")
        data = self.load_combined_results(combined_results_file)
        all_results = data.get('results', {})

        if not all_results:
            logger.error("No results found to synthesize")
            return None

        # Calculate consensus metrics
        logger.info("\n2. Calculating consensus metrics...")
        metrics = self.calculate_consensus_metrics(all_results)

        # Extract recommendations
        logger.info("\n3. Extracting recommendations...")
        recommendations = self.extract_key_recommendations(all_results)

        # Generate synthesis
        logger.info("\n4. Generating comprehensive synthesis...")
        synthesis = self.generate_synthesis(all_results)

        # Create final report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_file': combined_results_file,
                'num_models': len(all_results)
            },
            'consensus_metrics': metrics,
            'categorized_recommendations': recommendations,
            'claude_synthesis': synthesis,
            'individual_analyses': all_results
        }

        # Save report
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'validation_results/final_validation_report_{timestamp}.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\n5. Report saved to {output_file}")

        # Print summary
        self.print_summary(report)

        return output_file

    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable summary"""

        logger.info("\n" + "="*60)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("="*60)

        metrics = report['consensus_metrics']

        logger.info(f"\nOverall Score: {metrics.get('mean_score', 'N/A'):.2f} ± {metrics.get('std_score', 'N/A'):.2f}")
        logger.info(f"Score Range: {metrics.get('min_score', 'N/A'):.2f} - {metrics.get('max_score', 'N/A'):.2f}")

        logger.info(f"\nHigh Agreement Dimensions ({len(metrics.get('agreements', []))}):")
        for item in metrics.get('agreements', [])[:5]:
            logger.info(f"  - {item['dimension']}: {item['mean_score']:.2f}")

        logger.info(f"\nHigh Disagreement Dimensions ({len(metrics.get('disagreements', []))}):")
        for item in metrics.get('disagreements', [])[:5]:
            logger.info(f"  - {item['dimension']}: std={item['std']:.2f}")

        logger.info("\nClaude's Synthesis:")
        logger.info("-" * 60)
        synthesis_preview = report['claude_synthesis'][:500] + "..." if len(report['claude_synthesis']) > 500 else report['claude_synthesis']
        logger.info(synthesis_preview)
        logger.info("-" * 60)


def main():
    """Main execution function"""

    import glob

    # Find most recent combined results file
    results_files = glob.glob('validation_results/combined_analysis_*.json')

    if not results_files:
        logger.error("No combined analysis results found. Run manuscript_validator.py first.")
        return

    latest_file = max(results_files, key=os.path.getctime)
    logger.info(f"Using latest results file: {latest_file}")

    # Create synthesis
    engine = SynthesisEngine()
    report_file = engine.create_validation_report(latest_file)

    logger.info("\n" + "="*60)
    logger.info("SYNTHESIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Final validation report: {report_file}")
    logger.info("\nYou can now review the complete validation in the JSON report.")


if __name__ == "__main__":
    main()

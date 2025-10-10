"""
Structured prompt templates for manuscript validation
"""

from typing import Dict, Any


def get_prompt_for_dimension(dimension: str, context: Dict[str, Any]) -> str:
    """Generate structured prompt for specific analysis dimension"""

    manuscript_summary = context.get('manuscript_summary', '')
    evaluation_results = context.get('evaluation_results', {})
    datasets_info = context.get('datasets_info', {})

    base_context = f"""
You are a peer reviewer for an academic journal evaluating a research manuscript on topic modeling evaluation.

RESEARCH SUMMARY:
{manuscript_summary}

EVALUATION RESULTS:
{format_results(evaluation_results)}

DATASETS:
{format_datasets(datasets_info)}
"""

    dimension_prompts = {
        "methodology_validity": f"""
{base_context}

TASK: Evaluate the METHODOLOGY VALIDITY

Assess:
1. Are the chosen metrics (NPMI, C_v coherence, KLD, JSD, IRBO) appropriate for topic model evaluation?
2. Is the combination of statistical and LLM-based evaluation justified?
3. Are there missing metrics that should be included?
4. Does the methodology align with current best practices in topic modeling?

Provide your analysis in this format:
SCORE: <0-10, where 10=excellent methodology>
CONFIDENCE: <0-1, your confidence in this assessment>
REASONING: <detailed explanation>
RECOMMENDATIONS:
- <specific actionable recommendation 1>
- <specific actionable recommendation 2>
- <etc.>
""",

        "statistical_rigor": f"""
{base_context}

TASK: Evaluate the STATISTICAL RIGOR

Assess:
1. Are the normalization methods (0-1 scaling) mathematically sound?
2. Is the weighting scheme for overall score justified?
3. Are statistical assumptions clearly stated and validated?
4. Are there potential biases in the evaluation metrics?
5. Is the sample size (number of topics, documents) adequate?

Provide your analysis in this format:
SCORE: <0-10, where 10=highly rigorous>
CONFIDENCE: <0-1, your confidence in this assessment>
REASONING: <detailed explanation>
RECOMMENDATIONS:
- <specific actionable recommendation 1>
- <specific actionable recommendation 2>
- <etc.>
""",

        "experimental_design": f"""
{base_context}

TASK: Evaluate the EXPERIMENTAL DESIGN

Assess:
1. Is the use of three datasets (distinct, similar, more_similar) well-motivated?
2. Do these datasets provide adequate contrast for validation?
3. Are there confounding variables not controlled for?
4. Is the experimental setup reproducible?
5. Are baselines or comparisons adequate?

Provide your analysis in this format:
SCORE: <0-10, where 10=excellent design>
CONFIDENCE: <0-1, your confidence in this assessment>
REASONING: <detailed explanation>
RECOMMENDATIONS:
- <specific actionable recommendation 1>
- <specific actionable recommendation 2>
- <etc.>
""",

        "results_interpretation": f"""
{base_context}

TASK: Evaluate the RESULTS INTERPRETATION

Assess:
1. Do the conclusions logically follow from the results?
2. Are alternative explanations considered?
3. Are limitations acknowledged?
4. Is statistical significance properly reported?
5. Are visualizations and tables clear and accurate?

Provide your analysis in this format:
SCORE: <0-10, where 10=excellent interpretation>
CONFIDENCE: <0-1, your confidence in this assessment>
REASONING: <detailed explanation>
RECOMMENDATIONS:
- <specific actionable recommendation 1>
- <specific actionable recommendation 2>
- <etc.>
""",

        "contribution_assessment": f"""
{base_context}

TASK: Evaluate the RESEARCH CONTRIBUTION

Assess:
1. What is novel about this work compared to existing literature?
2. Does it advance the field of topic modeling evaluation?
3. Are the contributions clearly stated?
4. Is this work of sufficient quality for publication in a peer-reviewed journal?
5. Who would benefit from this research?

Provide your analysis in this format:
SCORE: <0-10, where 10=highly significant contribution>
CONFIDENCE: <0-1, your confidence in this assessment>
REASONING: <detailed explanation>
RECOMMENDATIONS:
- <specific actionable recommendation 1>
- <specific actionable recommendation 2>
- <etc.>
""",

        "limitations_identification": f"""
{base_context}

TASK: Identify LIMITATIONS and POTENTIAL ISSUES

Assess:
1. What are the main limitations of this approach?
2. Are there validity threats not addressed?
3. What assumptions might not hold in practice?
4. Are there ethical considerations?
5. What future work is needed?

Provide your analysis in this format:
SCORE: <0-10, where 10=comprehensive limitation discussion>
CONFIDENCE: <0-1, your confidence in this assessment>
REASONING: <detailed explanation>
RECOMMENDATIONS:
- <specific actionable recommendation 1>
- <specific actionable recommendation 2>
- <etc.>
"""
    }

    return dimension_prompts.get(dimension, "")


def format_results(results: Dict[str, Any]) -> str:
    """Format evaluation results for prompt"""
    if not results:
        return "No evaluation results provided"

    formatted = []
    for dataset_name, metrics in results.items():
        formatted.append(f"\n{dataset_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted.append(f"  {metric_name}: {value:.4f}")
            else:
                formatted.append(f"  {metric_name}: {value}")

    return "\n".join(formatted)


def format_datasets(datasets: Dict[str, Any]) -> str:
    """Format dataset information for prompt"""
    if not datasets:
        return "No dataset information provided"

    formatted = []
    for name, info in datasets.items():
        formatted.append(f"\n{name}:")
        formatted.append(f"  Description: {info.get('description', 'N/A')}")
        formatted.append(f"  Number of topics: {info.get('num_topics', 'N/A')}")
        formatted.append(f"  Number of documents: {info.get('num_documents', 'N/A')}")

    return "\n".join(formatted)


def get_synthesis_prompt(all_analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate prompt for final synthesis across all LLM analyses"""

    analyses_summary = []
    for model_name, analysis in all_analyses.items():
        overall = analysis.get('overall_assessment', {})
        analyses_summary.append(f"""
{model_name}:
  Overall Score: {overall.get('overall_score', 'N/A')}
  Recommendation: {overall.get('recommendation', 'N/A')}
  Confidence: {overall.get('avg_confidence', 'N/A')}
""")

    return f"""
You are synthesizing analyses from multiple LLMs evaluating a research manuscript on topic modeling.

INDIVIDUAL LLM ASSESSMENTS:
{''.join(analyses_summary)}

DETAILED ANALYSES:
{format_detailed_analyses(all_analyses)}

TASK: Provide a comprehensive synthesis that:
1. Identifies points of CONSENSUS across LLMs
2. Identifies points of DISAGREEMENT and explains why
3. Weighs evidence to make final recommendations
4. Provides actionable next steps for the authors
5. Assigns an overall confidence level

Format your synthesis as:
CONSENSUS POINTS:
- <point 1>
- <point 2>
...

DISAGREEMENTS:
- <disagreement 1 with explanation>
- <disagreement 2 with explanation>
...

FINAL VERDICT:
<Overall assessment>

PUBLICATION RECOMMENDATION: <READY|MINOR_REVISIONS|MAJOR_REVISIONS|REJECT>

CONFIDENCE: <0-1>

ACTIONABLE RECOMMENDATIONS:
1. <Priority recommendation>
2. <Second priority>
...
"""


def format_detailed_analyses(analyses: Dict[str, Dict[str, Any]]) -> str:
    """Format detailed analyses for synthesis prompt"""
    formatted = []

    for model_name, analysis in analyses.items():
        formatted.append(f"\n=== {model_name} ===")
        for dim, details in analysis.get('analyses', {}).items():
            formatted.append(f"\n{dim}:")
            formatted.append(f"  Score: {details.get('score', 'N/A')}")
            formatted.append(f"  Reasoning: {details.get('reasoning', 'N/A')[:200]}...")

    return "\n".join(formatted)

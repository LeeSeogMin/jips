"""
Run all 4 LLM validators to check reviewer comments
Uses: Anthropic Claude, OpenAI GPT-4.1, Google Gemini, xAI Grok
"""

from manuscript_reviewer_validator import ReviewerCommentValidator
from anthropic_analyzer import AnthropicAnalyzer
from openai_analyzer import OpenAIAnalyzer
from gemini_analyzer import GeminiAnalyzer
from grok_analyzer import GrokAnalyzer
from pathlib import Path
import json
from datetime import datetime


def create_llm_validation_prompt(failed_checks: list) -> str:
    """Create prompt for LLM to validate manuscript"""
    prompt = """You are an expert peer reviewer. A manuscript has been revised based on reviewer comments.

VALIDATION RESULTS:
"""

    if not failed_checks:
        prompt += "\n✅ All automated checks passed (21/21)\n"
        prompt += "\nPlease provide a final assessment:\n"
        prompt += "1. Overall quality (1-10 score)\n"
        prompt += "2. Key strengths (3 points)\n"
        prompt += "3. Any remaining concerns\n"
        prompt += "4. Publication recommendation (ACCEPT/MINOR_REVISIONS/MAJOR_REVISIONS/REJECT)\n"
    else:
        prompt += f"\n⚠️ {len(failed_checks)} checks failed:\n\n"
        for check in failed_checks:
            prompt += f"- {check['item']}\n"
            prompt += f"  Missing: {', '.join(check['missing_values'])}\n\n"

        prompt += "\nPlease assess:\n"
        prompt += "1. Severity of missing items (CRITICAL/MODERATE/MINOR)\n"
        prompt += "2. Impact on publication readiness\n"
        prompt += "3. Specific recommendations for fixes\n"
        prompt += "4. Publication recommendation\n"

    prompt += "\nFormat your response as:\n"
    prompt += "SCORE: <1-10>\n"
    prompt += "SEVERITY: <CRITICAL/MODERATE/MINOR>\n"
    prompt += "ASSESSMENT: <detailed assessment>\n"
    prompt += "RECOMMENDATIONS: <specific actions>\n"
    prompt += "DECISION: <ACCEPT/MINOR_REVISIONS/MAJOR_REVISIONS/REJECT>\n"

    return prompt


def run_all_validators(manuscript_path: str):
    """Run all 4 LLM validators"""
    print("="*70)
    print("MULTI-LLM REVIEWER VALIDATION")
    print("="*70)
    print(f"Manuscript: {Path(manuscript_path).name}")
    print(f"LLMs: Anthropic Claude, OpenAI GPT-4.1, Google Gemini, xAI Grok")
    print("="*70)

    # Step 1: Run automated validation
    print("\n[1/5] Running automated validation...")
    validator = ReviewerCommentValidator(manuscript_path)
    results = validator.validate_all()

    # Collect failed checks
    failed_checks = []
    for issue_type in ['major_issues', 'minor_issues']:
        for issue_id, issue_data in results[issue_type].items():
            for check in issue_data['checks']:
                if check['status'] == 'fail':
                    failed_checks.append(check)

    print(f"\nAutomated validation: {results['summary']['passed']}/{results['summary']['total_checks']} passed")

    # Step 2: Prepare LLM prompt
    prompt = create_llm_validation_prompt(failed_checks)

    # Step 3-6: Run each LLM
    llm_results = {}

    analyzers = [
        ("Anthropic Claude Sonnet 4.5", AnthropicAnalyzer()),
        ("OpenAI GPT-4.1", OpenAIAnalyzer()),
        ("Google Gemini 2.5", GeminiAnalyzer()),
        ("xAI Grok", GrokAnalyzer())
    ]

    for i, (name, analyzer) in enumerate(analyzers, start=2):
        print(f"\n[{i}/5] Running {name} validation...")
        try:
            response = analyzer._call_llm(prompt, temperature=0.0)
            llm_results[name] = {
                "success": True,
                "response": response,
                "parsed": parse_llm_response(response)
            }
            print(f"✅ {name} validation complete")
        except Exception as e:
            print(f"❌ {name} validation failed: {e}")
            llm_results[name] = {
                "success": False,
                "error": str(e)
            }

    # Aggregate results
    print("\n" + "="*70)
    print("MULTI-LLM VALIDATION RESULTS")
    print("="*70)

    scores = []
    decisions = []

    for name, result in llm_results.items():
        if result['success'] and 'parsed' in result:
            parsed = result['parsed']
            print(f"\n{name}:")
            print(f"  Score: {parsed.get('score', 'N/A')}/10")
            print(f"  Decision: {parsed.get('decision', 'N/A')}")
            print(f"  Severity: {parsed.get('severity', 'N/A')}")

            if parsed.get('score'):
                scores.append(parsed['score'])
            if parsed.get('decision'):
                decisions.append(parsed['decision'])

    # Consensus
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n{'='*70}")
        print(f"CONSENSUS RESULTS")
        print(f"{'='*70}")
        print(f"Average Score: {avg_score:.2f}/10")
        print(f"Decisions: {', '.join(decisions)}")

        if avg_score >= 8.0:
            print(f"\n🎉 STRONG CONSENSUS: Manuscript ready for publication")
        elif avg_score >= 7.0:
            print(f"\n✅ CONSENSUS: Minor revisions recommended")
        elif avg_score >= 5.0:
            print(f"\n⚠️ MIXED: Major revisions needed")
        else:
            print(f"\n❌ CONCERN: Significant issues remain")

    # Save results
    output = {
        "manuscript": Path(manuscript_path).name,
        "timestamp": datetime.now().isoformat(),
        "automated_validation": {
            "passed": results['summary']['passed'],
            "failed": results['summary']['failed'],
            "total": results['summary']['total_checks'],
            "failed_checks": failed_checks
        },
        "llm_validations": llm_results,
        "consensus": {
            "average_score": sum(scores) / len(scores) if scores else None,
            "decisions": decisions,
            "recommendation": get_consensus_recommendation(scores, decisions) if scores else None
        }
    }

    output_path = Path(manuscript_path).parent / f"multi_llm_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Full results saved: {output_path.name}")
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    return output


def parse_llm_response(response: str) -> dict:
    """Parse structured LLM response"""
    parsed = {}

    lines = response.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("SCORE:"):
            try:
                parsed['score'] = float(line.split(":", 1)[1].strip().split('/')[0])
            except:
                pass
        elif line.startswith("SEVERITY:"):
            parsed['severity'] = line.split(":", 1)[1].strip()
        elif line.startswith("ASSESSMENT:"):
            current_section = 'assessment'
            parsed['assessment'] = line.split(":", 1)[1].strip()
        elif line.startswith("RECOMMENDATIONS:"):
            current_section = 'recommendations'
            parsed['recommendations'] = line.split(":", 1)[1].strip()
        elif line.startswith("DECISION:"):
            parsed['decision'] = line.split(":", 1)[1].strip()
        elif current_section and line:
            parsed[current_section] += " " + line

    return parsed


def get_consensus_recommendation(scores: list, decisions: list) -> str:
    """Get consensus recommendation from all LLMs"""
    if not scores:
        return "UNCERTAIN"

    avg_score = sum(scores) / len(scores)

    # Count decisions
    accept_count = decisions.count("ACCEPT")
    minor_count = decisions.count("MINOR_REVISIONS")
    major_count = decisions.count("MAJOR_REVISIONS")
    reject_count = decisions.count("REJECT")

    if avg_score >= 8.0 and accept_count >= 2:
        return "ACCEPT"
    elif avg_score >= 7.0 and (accept_count + minor_count) >= 3:
        return "MINOR_REVISIONS"
    elif avg_score >= 5.0:
        return "MAJOR_REVISIONS"
    else:
        return "REJECT"


def main():
    """Main execution"""
    import sys

    if len(sys.argv) > 1:
        manuscript_path = sys.argv[1]
    else:
        manuscript_path = r"C:\jips\docs\manuscript_F.docx"

    if not Path(manuscript_path).exists():
        print(f"❌ Error: Manuscript not found: {manuscript_path}")
        return

    run_all_validators(manuscript_path)


if __name__ == "__main__":
    main()

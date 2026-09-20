import argparse
import json
from pathlib import Path

from llm_report_agent import build_local_context


def collect_gap_detection_text(context):
    strategy = context["local_strategy"]
    agent = context["local_analysis"]["agent_assessment"]
    items = []
    items.extend(strategy.get("specific_questions", []))
    items.extend(agent.get("blockers", []))
    items.extend(agent.get("evidence_requests", []))
    items.extend(agent.get("interview_focus", []))
    return "\n".join(items).lower()


def evaluate_expected_gaps(context, expected_gap_data):
    detection_text = collect_gap_detection_text(context)
    results = []

    for gap in expected_gap_data["expected_gaps"]:
        matched_terms = [
            term
            for term in gap["match_terms"]
            if term.lower() in detection_text
        ]
        results.append({
            "id": gap["id"],
            "description": gap["description"],
            "detected": bool(matched_terms),
            "matched_terms": matched_terms,
        })

    detected_count = sum(result["detected"] for result in results)
    return {
        "detected": detected_count,
        "total": len(results),
        "score": detected_count / len(results) if results else 0,
        "results": results,
    }


def render_evaluation(context, evaluation):
    analysis = context["local_analysis"]
    strategy = context["local_strategy"]
    lines = [
        "SR&ED incomplete-intake capability evaluation",
        "=" * 80,
        f"Classification: {analysis['prediction']}",
        f"CRA alignment: {analysis['cra_alignment']}",
        f"Structure: {strategy['selected_structure']['mode']}",
        "Streams: " + ", ".join(stream["id"] for stream in strategy["streams"]),
        (
            f"Deliberate gaps detected: {evaluation['detected']}/"
            f"{evaluation['total']} ({evaluation['score']:.0%})"
        ),
        "",
        "Expected gap results",
        "-" * 80,
    ]

    for result in evaluation["results"]:
        status = "PASS" if result["detected"] else "MISS"
        matched = ", ".join(result["matched_terms"]) or "none"
        lines.append(f"{status}: {result['id']} - matched: {matched}")
        lines.append(f"  {result['description']}")

    lines.extend(["", "Generated specific questions", "-" * 80])
    for index, question in enumerate(strategy["specific_questions"], start=1):
        lines.append(f"{index}. {question}")

    lines.extend(["", "Detected blockers", "-" * 80])
    lines.extend(
        f"- {blocker}"
        for blocker in analysis["agent_assessment"]["blockers"]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether local analysis detects deliberate intake gaps."
    )
    parser.add_argument("input_path", help="Incomplete project information draft.")
    parser.add_argument("expected_gaps_path", help="JSON answer key of deliberate gaps.")
    args = parser.parse_args()

    source_text = Path(args.input_path).read_text(encoding="utf-8")
    expected_gap_data = json.loads(
        Path(args.expected_gaps_path).read_text(encoding="utf-8")
    )
    context = build_local_context(source_text)
    evaluation = evaluate_expected_gaps(context, expected_gap_data)
    print(render_evaluation(context, evaluation))


if __name__ == "__main__":
    main()

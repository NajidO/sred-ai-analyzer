import argparse
from datetime import datetime
from pathlib import Path

from analysis_engine import BASE_DIR
from case_store import load_case


TECHNICAL_REPORTS_DIR = "technical_reports"


def get_technical_report_dir(base_dir):
    return Path(base_dir) / TECHNICAL_REPORTS_DIR


def generate_technical_report_for_case(case_reference, base_dir=BASE_DIR, timestamp=None):
    case_data = load_case(case_reference, base_dir)
    return save_technical_report(case_data, base_dir, timestamp=timestamp)


def save_technical_report(case_data, base_dir=BASE_DIR, timestamp=None):
    report_dir = get_technical_report_dir(base_dir)
    report_dir.mkdir(exist_ok=True)

    report = build_technical_report(case_data)
    report_timestamp = timestamp or datetime.now().astimezone()
    case_id = sanitize_filename(case_data.get("case_id", "case"))
    file_timestamp = report_timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"{case_id}_technical_report_{file_timestamp}.md"
    report_path.write_text(render_technical_report(report), encoding="utf-8")

    return report_path


def build_technical_report(case_data):
    final_assessment = case_data["final_assessment"]
    agent_assessment = final_assessment["agent_assessment"]
    cra_checks = final_assessment["cra_checks"]
    evidence_map = final_assessment["evidence_map"]
    readiness = assess_report_readiness(final_assessment)

    return {
        "title": "SR&ED Technical Report Draft",
        "case_id": case_data.get("case_id", ""),
        "created_at": case_data.get("created_at", ""),
        "status": case_data.get("status", ""),
        "source_case_path": case_data.get("_path", ""),
        "readiness": readiness,
        "metadata": build_metadata(case_data, final_assessment, agent_assessment),
        "sections": [
            build_executive_summary(final_assessment, agent_assessment, readiness),
            build_project_overview(case_data),
            build_technical_objective_section(cra_checks, final_assessment),
            build_uncertainty_section(cra_checks, evidence_map),
            build_standard_practice_section(final_assessment),
            build_investigation_section(cra_checks, evidence_map, case_data),
            build_results_section(cra_checks, evidence_map),
            build_advancement_section(cra_checks),
            build_evidence_section(cra_checks, agent_assessment, evidence_map),
            build_gaps_section(agent_assessment),
        ],
    }


def build_metadata(case_data, final_assessment, agent_assessment):
    probability_map = final_assessment.get("category_probabilities", {})

    return {
        "working_classification": final_assessment.get("prediction", ""),
        "classification_confidence": format_confidence(probability_map),
        "cra_alignment": final_assessment.get("cra_alignment", ""),
        "case_stage": agent_assessment.get("case_stage", ""),
        "priority": agent_assessment.get("priority", ""),
        "decision": agent_assessment.get("decision", ""),
        "parent_case_id": case_data.get("parent_case_id", ""),
    }


def build_executive_summary(final_assessment, agent_assessment, readiness):
    prediction = final_assessment.get("prediction", "")
    alignment = final_assessment.get("cra_alignment", "")
    decision = agent_assessment.get("decision", "")

    body = [
        (
            f"The current working classification is `{prediction}` with CRA checklist "
            f"alignment of `{alignment}`."
        ),
        (
            f"Report readiness is `{readiness['level']}` with a checklist score of "
            f"{readiness['score']}%."
        ),
        decision,
        (
            "This draft is for analyst review. It should be treated as a structured "
            "starting point, not as a final eligibility opinion."
        ),
    ]

    return {
        "heading": "Executive Summary",
        "body": body,
        "bullets": readiness["reasons"],
        "gaps": readiness["gaps"],
    }


def build_project_overview(case_data):
    body = [
        "The current project record is based on the saved intake description below.",
        blockquote(case_data.get("updated_text", "")),
    ]

    original_text = case_data.get("original_text", "").strip()
    updated_text = case_data.get("updated_text", "").strip()
    if original_text and original_text != updated_text:
        body.append("Original intake description:")
        body.append(blockquote(original_text))

    return {
        "heading": "Project Overview",
        "body": body,
        "bullets": build_answer_bullets(case_data.get("answers", [])),
        "gaps": [],
    }


def build_technical_objective_section(cra_checks, final_assessment):
    advancement_check = cra_checks["technological_advancement"]

    body = [
        "Working technical objective:",
        (
            "Define the capability, performance threshold, reliability target, or "
            "technical limitation the team was trying to advance."
        ),
        f"Current analyzer finding: {advancement_check['comment']}",
    ]

    gaps = []
    if advancement_check["status"] != "present":
        gaps.append(
            "Clarify the specific technological capability or knowledge the project sought to advance."
        )

    explanation = final_assessment.get("explanation", {})
    reasons = explanation.get("reasons", [])

    return {
        "heading": "Technical Objective",
        "body": body,
        "bullets": reasons[:3],
        "gaps": gaps,
    }


def build_uncertainty_section(cra_checks, evidence_map):
    uncertainty_check = cra_checks["technological_uncertainty"]

    body = [
        "Working uncertainty statement:",
        evidence_map.get("possible_uncertainty", "No uncertainty statement available."),
        f"Current analyzer finding: {uncertainty_check['comment']}",
    ]

    gaps = []
    if uncertainty_check["status"] != "present":
        gaps.append(
            "State why the uncertainty could not be resolved using standard practice, known tools, or vendor documentation."
        )

    return {
        "heading": "Technological Uncertainty",
        "body": body,
        "bullets": [],
        "gaps": gaps,
    }


def build_standard_practice_section(final_assessment):
    signals = final_assessment.get("signals", {})
    routine_signals = signals.get("routine_signals", [])

    body = [
        (
            "Describe the known methods, standard tools, vendor features, accepted "
            "engineering patterns, or routine configurations that were considered first."
        )
    ]

    if routine_signals:
        body.append(
            "The analyzer detected routine implementation signals that should be separated from eligible experimental work."
        )
    else:
        body.append(
            "No strong routine implementation signals were detected, but the standard-practice baseline should still be documented."
        )

    gaps = [
        "Identify which standard approaches were tried, why they were insufficient, and what limitation remained unresolved."
    ]

    return {
        "heading": "Standard Practice Baseline",
        "body": body,
        "bullets": routine_signals,
        "gaps": gaps,
    }


def build_investigation_section(cra_checks, evidence_map, case_data):
    investigation_check = cra_checks["systematic_investigation"]
    experiments = evidence_map.get("possible_experiments", [])

    body = [
        "Summarize the systematic investigation through experiments, analysis, prototypes, benchmarks, or iterations.",
        f"Current analyzer finding: {investigation_check['comment']}",
    ]

    gaps = []
    if investigation_check["status"] != "present":
        gaps.append(
            "Add a timeline of hypotheses, tests, iterations, failed attempts, and decision points."
        )

    answer_bullets = build_answer_bullets(case_data.get("answers", []))

    return {
        "heading": "Systematic Investigation",
        "body": body,
        "bullets": experiments + answer_bullets,
        "gaps": gaps,
    }


def build_results_section(cra_checks, evidence_map):
    results_check = cra_checks["experimental_results"]

    body = [
        "Working results summary:",
        evidence_map.get("possible_results", "No results summary available."),
        f"Current analyzer finding: {results_check['comment']}",
    ]

    gaps = []
    if results_check["status"] != "present":
        gaps.append(
            "Document measured results for each approach, including failures and abandoned approaches."
        )

    return {
        "heading": "Results And Technical Learning",
        "body": body,
        "bullets": [],
        "gaps": gaps,
    }


def build_advancement_section(cra_checks):
    advancement_check = cra_checks["technological_advancement"]

    body = [
        "Explain the technological knowledge gained from the investigation, not just the business outcome or finished feature.",
        f"Current analyzer finding: {advancement_check['comment']}",
    ]

    gaps = []
    if advancement_check["status"] != "present":
        gaps.append(
            "Connect the test results to a technological advancement or learning that was not available at the start."
        )

    return {
        "heading": "Technological Advancement",
        "body": body,
        "bullets": [],
        "gaps": gaps,
    }


def build_evidence_section(cra_checks, agent_assessment, evidence_map):
    evidence_check = cra_checks["supporting_evidence"]
    evidence_requests = agent_assessment.get("evidence_requests", [])
    missing_evidence = evidence_map.get("missing_evidence", [])

    body = [
        f"Current analyzer finding: {evidence_check['comment']}",
        "Collect evidence that ties the uncertainty, experiments, results, and learning to contemporaneous records.",
    ]

    gaps = []
    if evidence_check["status"] != "present":
        gaps.append(
            "Attach or reference supporting records before treating the technical report as claim-ready."
        )

    return {
        "heading": "Supporting Evidence Inventory",
        "body": body,
        "bullets": unique_items(evidence_requests + missing_evidence),
        "gaps": gaps,
    }


def build_gaps_section(agent_assessment):
    return {
        "heading": "Open Gaps And Analyst Questions",
        "body": [
            "Use these items to drive the next interview or evidence request cycle."
        ],
        "bullets": unique_items(
            agent_assessment.get("blockers", [])
            + agent_assessment.get("interview_focus", [])
            + agent_assessment.get("action_plan", [])
        ),
        "gaps": [],
    }


def assess_report_readiness(final_assessment):
    checks = final_assessment["cra_checks"]
    statuses = [result["status"] for result in checks.values()]
    present_count = statuses.count("present")
    partial_count = statuses.count("partial")
    missing_count = statuses.count("missing")
    max_score = max(len(statuses) * 2, 1)
    score = round(((present_count * 2) + partial_count) / max_score * 100)
    prediction = final_assessment.get("prediction", "")
    agent_assessment = final_assessment.get("agent_assessment", {})
    case_stage = agent_assessment.get("case_stage", "")

    if prediction == "routine" and case_stage == "routine_screening":
        level = "not_report_ready"
        reasons = ["The work currently appears routine rather than SR&ED candidate work."]
    elif score >= 80 and missing_count == 0:
        level = "draft_ready"
        reasons = ["Most technical report elements are present or partially supported."]
    elif score >= 50:
        level = "draft_with_gaps"
        reasons = ["A report draft can be prepared, but key gaps need analyst follow-up."]
    else:
        level = "intake_required"
        reasons = ["The case record is not detailed enough for a reliable technical report draft."]

    gaps = [
        f"{check_name}: {result['comment']}"
        for check_name, result in checks.items()
        if result["status"] != "present"
    ]

    return {
        "level": level,
        "score": score,
        "present_count": present_count,
        "partial_count": partial_count,
        "missing_count": missing_count,
        "reasons": reasons,
        "gaps": gaps,
    }


def render_technical_report(report):
    lines = [
        f"# {report['title']}",
        "",
        f"Case ID: {report['case_id']}",
        f"Case created: {report['created_at']}",
        f"Case status: {report['status']}",
    ]

    if report["source_case_path"]:
        lines.append(f"Source case file: {report['source_case_path']}")

    lines.extend([
        "",
        "## Report Metadata",
    ])

    for label, value in report["metadata"].items():
        if value:
            lines.append(f"- {titleize(label)}: {value}")

    lines.extend([
        f"- Readiness: {report['readiness']['level']}",
        f"- Readiness score: {report['readiness']['score']}%",
        "",
    ])

    for section in report["sections"]:
        lines.append(f"## {section['heading']}")
        lines.append("")

        for paragraph in section.get("body", []):
            if paragraph:
                lines.append(paragraph)
                lines.append("")

        bullets = section.get("bullets", [])
        if bullets:
            for bullet in bullets:
                lines.append(f"- {bullet}")
            lines.append("")

        gaps = section.get("gaps", [])
        if gaps:
            lines.append("Gaps to close:")
            for gap in gaps:
                lines.append(f"- {gap}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_answer_bullets(answers):
    bullets = []

    for item in answers:
        answer = item.get("answer", "").strip()
        question = item.get("question", "").strip()
        if answer:
            if question:
                bullets.append(f"{question} Answer: {answer}")
            else:
                bullets.append(answer)

    return bullets


def blockquote(text):
    stripped_text = text.strip()

    if not stripped_text:
        return "> No project description has been captured yet."

    return "\n".join(f"> {line}" if line else ">" for line in stripped_text.splitlines())


def format_confidence(probability_map):
    if not probability_map:
        return ""

    return f"{max(probability_map.values()):.2f}"


def titleize(value):
    return value.replace("_", " ").title()


def sanitize_filename(value):
    cleaned = [
        character
        for character in str(value)
        if character.isalnum() or character in {"-", "_"}
    ]
    return "".join(cleaned) or "case"


def unique_items(items):
    seen = set()
    unique = []

    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate an SR&ED technical report draft from a saved case file.",
    )
    parser.add_argument("case_id", help="Case ID, file name, or JSON path.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the generated report after saving it.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    report_path = generate_technical_report_for_case(args.case_id)

    print("Technical report saved:")
    print(report_path)

    if args.show:
        print()
        print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

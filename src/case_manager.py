import argparse

from analysis_engine import BASE_DIR, MODEL_PATH, analyze_text, load_classifier
from case_store import (
    list_case_files,
    load_case,
    save_intake_case_file,
    summarize_case_for_listing,
)
from intake_agent import (
    collect_intake_answers,
    print_case_snapshot,
    print_section,
    save_analysis_report,
    select_intake_questions,
)


def list_cases(base_dir=BASE_DIR):
    cases = list_case_files(base_dir)

    if not cases:
        print("No saved case files found.")
        return []

    print("Saved SR&ED case files")
    print("=" * 80)

    for case_data in cases:
        summary = summarize_case_for_listing(case_data)
        print(f"\nCase ID: {summary['case_id']}")
        print(f"Created: {summary['created_at']}")
        print(f"Status: {summary['status']}")
        print(f"Prediction: {summary['prediction']}")
        print(f"Stage: {summary['case_stage']}")
        print(f"Priority: {summary['priority']}")
        print(f"Summary: {summary['summary']}")
        print(f"Path: {summary['path']}")

    return cases


def inspect_case(case_reference, base_dir=BASE_DIR):
    case_data = load_case(case_reference, base_dir)
    final_assessment = case_data["final_assessment"]
    agent_assessment = final_assessment["agent_assessment"]

    print(f"Case ID: {case_data['case_id']}")
    print(f"Created: {case_data['created_at']}")
    print(f"Status: {case_data['status']}")
    if case_data.get("parent_case_id"):
        print(f"Parent case: {case_data['parent_case_id']}")

    print_section("CURRENT ASSESSMENT")
    print(f"Prediction: {final_assessment['prediction']}")
    print(f"CRA alignment: {final_assessment['cra_alignment']}")
    print(f"Case stage: {agent_assessment['case_stage']}")
    print(f"Priority: {agent_assessment['priority']}")
    print(f"Decision: {agent_assessment['decision']}")

    print("\nBlockers:")
    for blocker in agent_assessment["blockers"]:
        print(f"- {blocker}")

    print("\nEvidence to request:")
    for item in agent_assessment["evidence_requests"]:
        print(f"- {item}")

    print("\nInterview focus:")
    for question in agent_assessment["interview_focus"]:
        print(f"- {question}")

    print_section("UPDATED CASE DESCRIPTION")
    print(case_data["updated_text"])

    print_section("REPORT")
    print(case_data["report_path"])

    print_section("CASE FILE")
    print(case_data["_path"])

    return case_data


def resume_case(case_reference, model, base_dir=BASE_DIR, max_questions=5):
    case_data = load_case(case_reference, base_dir)
    final_assessment = case_data["final_assessment"]
    questions = select_intake_questions(
        final_assessment["agent_assessment"],
        max_questions=max_questions,
    )

    print(f"Resuming case: {case_data['case_id']}")
    print_case_summary_for_resume(case_data)

    print_section("FOLLOW-UP QUESTIONS")
    if not questions:
        print("No follow-up questions were generated.")
        answers = []
    else:
        answers = collect_intake_answers(questions)

    if not answers:
        print("\nNo new answers captured. Case was not updated.")
        return {
            "case_data": case_data,
            "answers": [],
            "case_file_path": None,
        }

    updated_text = build_resumed_description(case_data["updated_text"], answers)
    final_analysis = analyze_text(updated_text, model)
    report_path = save_analysis_report(updated_text, final_analysis, base_dir=base_dir)
    initial_analysis = analyze_text(case_data["original_text"], model)
    session = {
        "initial_analysis": initial_analysis,
        "questions": questions,
        "answers": answers,
        "updated_text": updated_text,
        "final_analysis": final_analysis,
        "report_path": report_path,
        "parent_case_id": case_data["case_id"],
        "status": "resumed",
    }
    case_file_path = save_intake_case_file(session, base_dir)

    print_case_snapshot("UPDATED CASE ASSESSMENT", final_analysis)

    print_section("UPDATED CASE DESCRIPTION")
    print(updated_text)

    print_section("REPORT SAVED")
    print(report_path)

    print_section("CASE FILE SAVED")
    print(case_file_path)

    return {
        "case_data": case_data,
        "answers": answers,
        "updated_text": updated_text,
        "final_analysis": final_analysis,
        "report_path": report_path,
        "case_file_path": case_file_path,
    }


def build_resumed_description(current_text, answers):
    answered_items = [
        item
        for item in answers
        if item["answer"].strip()
    ]

    if not answered_items:
        return current_text.strip()

    lines = [
        current_text.strip(),
        "",
        "Additional follow-up intake answers:",
    ]

    for index, item in enumerate(answered_items, start=1):
        lines.append(f"{index}. Question: {item['question']}")
        lines.append(f"   Answer: {item['answer'].strip()}")

    return "\n".join(lines)


def print_case_summary_for_resume(case_data):
    final_assessment = case_data["final_assessment"]
    agent_assessment = final_assessment["agent_assessment"]

    print_section("CURRENT CASE SNAPSHOT")
    print(f"Prediction: {final_assessment['prediction']}")
    print(f"Case stage: {agent_assessment['case_stage']}")
    print(f"Priority: {agent_assessment['priority']}")
    print(f"Decision: {agent_assessment['decision']}")

    print("\nTop blockers:")
    for blocker in agent_assessment["blockers"][:3]:
        print(f"- {blocker}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="List, inspect, or resume saved SR&ED intake case files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List saved case files.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved case file.")
    inspect_parser.add_argument("case_id", help="Case ID, file name, or JSON path.")

    resume_parser = subparsers.add_parser("resume", help="Resume intake for a saved case file.")
    resume_parser.add_argument("case_id", help="Case ID, file name, or JSON path.")
    resume_parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum number of follow-up questions to ask.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        list_cases()
        return

    if args.command == "inspect":
        inspect_case(args.case_id)
        return

    if args.command == "resume":
        model = load_classifier(MODEL_PATH)
        resume_case(args.case_id, model, max_questions=args.max_questions)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

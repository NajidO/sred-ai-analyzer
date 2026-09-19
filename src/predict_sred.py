from pathlib import Path

import joblib

from rules import extract_signals
from questions import generate_followup_questions, generate_cra_reference_questions
from recommendation import generate_recommendation
from evidence_mapper import map_to_sred_framework
from report_writer import save_report
from cra_guideline_checker import check_against_cra_guidelines
from explanation import generate_label_explanation
from agent_assessment import build_agent_assessment

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "sred_classifier.joblib"

model = joblib.load(MODEL_PATH)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_section(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def analyze_project(text):
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    labels = model.classes_

    signals = extract_signals(text)
    questions = generate_followup_questions(prediction, signals)
    cra_reference_questions = generate_cra_reference_questions()
    recommendation = generate_recommendation(
        prediction,
        probabilities,
        labels,
        signals
    )
    evidence_map = map_to_sred_framework(text, signals)
    cra_check = check_against_cra_guidelines(text, signals)
    explanation = generate_label_explanation(
        prediction,
        probabilities,
        labels,
        signals,
        cra_check
    )
    agent_assessment = build_agent_assessment(
        prediction,
        probabilities,
        labels,
        signals,
        cra_check,
        evidence_map,
        questions,
        cra_reference_questions,
        explanation,
    )

    print_header("SR&ED TECHNICAL UNCERTAINTY ANALYSIS")

    print_section("PROJECT DESCRIPTION")
    print(text)

    print_section("TOP PREDICTION")
    print(prediction)

    print_section("CATEGORY PROBABILITIES")
    for label, probability in zip(labels, probabilities):
        print(f"{label}: {probability:.2f}")

    print_section("WHY THIS CLASSIFICATION")
    print(f"Model confidence in top prediction: {explanation['confidence']:.2f}")

    print("\nReasons:")
    for reason in explanation["reasons"]:
        print(f"- {reason}")

    print("\nCautions:")
    if explanation["cautions"]:
        for caution in explanation["cautions"]:
            print(f"- {caution}")
    else:
        print("- None")

    print("\nNext steps:")
    for step in explanation["next_steps"]:
        print(f"- {step}")

    print_section("AGENTIC CASE ASSESSMENT")
    print(f"Case stage: {agent_assessment['case_stage']}")
    print(f"Priority: {agent_assessment['priority']}")
    print(f"Decision: {agent_assessment['decision']}")
    print(f"Agent confidence: {agent_assessment['confidence']:.2f}")

    print("\nBlockers:")
    for blocker in agent_assessment["blockers"]:
        print(f"- {blocker}")

    print("\nEvidence to request:")
    for item in agent_assessment["evidence_requests"]:
        print(f"- {item}")

    print("\nInterview focus:")
    for question in agent_assessment["interview_focus"]:
        print(f"- {question}")

    print("\nAction plan:")
    for action in agent_assessment["action_plan"]:
        print(f"- {action}")

    print("\nHandoff summary:")
    print(agent_assessment["handoff_summary"])

    print_section("DETECTED SIGNALS")
    print("Uncertainty signals:")
    if signals["uncertainty_signals"]:
        for signal in signals["uncertainty_signals"]:
            print(f"- {signal}")
    else:
        print("- None detected")

    print("\nRoutine signals:")
    if signals["routine_signals"]:
        for signal in signals["routine_signals"]:
            print(f"- {signal}")
    else:
        print("- None detected")

    print_section("RECOMMENDATION")
    print(recommendation)

    print_section("SR&ED EVIDENCE MAP")

    print("\nPossible uncertainty:")
    print(evidence_map["possible_uncertainty"])

    print("\nPossible experiments:")
    for experiment in evidence_map["possible_experiments"]:
        print(f"- {experiment}")

    print("\nPossible results:")
    print(evidence_map["possible_results"])

    print("\nMissing evidence to request:")
    for item in evidence_map["missing_evidence"]:
        print(f"- {item}")

    print_section("CRA GUIDELINE ALIGNMENT CHECK")
    print("Overall alignment:")
    print(cra_check["overall_alignment"])

    print("\nChecklist:")
    for check_name, result in cra_check["checks"].items():
        print(f"\n{check_name}:")
        print(f"Status: {result['status']}")
        print(f"Comment: {result['comment']}")
    print_section("SUGGESTED FOLLOW-UP QUESTIONS")
    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")

    print_section("CRA-GROUNDED ENGAGEMENT QUESTIONS")
    for index, question in enumerate(cra_reference_questions, start=1):
        print(f"{index}. {question}")

    report_path = save_report(
        text,
        prediction,
        probabilities,
        labels,
        signals,
        explanation,
        agent_assessment,
        recommendation,
        evidence_map,
        cra_check,
        questions,
        cra_reference_questions,
        BASE_DIR
    )

    print_section("REPORT SAVED")
    print(report_path)


def main():
    print("\nSR&ED Technical Uncertainty Analyzer")
    print("Type a project description to analyze it.")
    print("Type 'quit' to stop.")

    while True:
        user_text = input("\nProject description: ")

        if user_text.lower() == "quit":
            print("\nGoodbye.")
            break

        if not user_text.strip():
            print("Please enter some text.")
            continue

        analyze_project(user_text)


if __name__ == "__main__":
    main()

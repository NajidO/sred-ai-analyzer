from pathlib import Path
from collections import Counter
from tempfile import TemporaryDirectory
import json

import pandas as pd
import joblib

import case_manager
from analysis_engine import analyze_text
from agent_assessment import build_agent_assessment
from case_manager import build_resumed_description, inspect_case, list_cases, resume_case
from cra_guideline_checker import check_against_cra_guidelines
from evidence_mapper import map_to_sred_framework
from explanation import generate_label_explanation
from intake_agent import build_updated_description, select_intake_questions
from case_store import (
    list_case_files,
    load_case,
    save_intake_case_file,
    summarize_case_for_listing,
)
from questions import generate_followup_questions, generate_cra_reference_questions
from rules import extract_signals
from technical_report import (
    assess_report_readiness,
    build_technical_report,
    generate_technical_report_for_case,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "sred_classifier.joblib"
TEST_PATH = BASE_DIR / "data" / "test_examples.csv"
TRAINING_PATH = BASE_DIR / "data" / "sred_training_data.csv"
EXPECTED_TRAINING_COLUMNS = ["text", "label"]
EXPECTED_TEST_COLUMNS = ["text", "expected_label"]
VALID_LABELS = {"routine", "borderline", "needs_more_info", "strong_sred"}


def validate_training_csv():
    training_df = pd.read_csv(TRAINING_PATH)

    if list(training_df.columns) != EXPECTED_TRAINING_COLUMNS:
        raise ValueError(
            f"Expected training columns {EXPECTED_TRAINING_COLUMNS}, "
            f"found {list(training_df.columns)}"
        )

    if training_df[EXPECTED_TRAINING_COLUMNS].isnull().any().any():
        raise ValueError("Training CSV contains blank text or label values.")

    invalid_labels = sorted(set(training_df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Training CSV contains invalid labels: {invalid_labels}")

    return training_df


def validate_test_examples_csv():
    test_df = pd.read_csv(TEST_PATH)

    if list(test_df.columns) != EXPECTED_TEST_COLUMNS:
        raise ValueError(
            f"Expected test columns {EXPECTED_TEST_COLUMNS}, "
            f"found {list(test_df.columns)}"
        )

    if test_df[EXPECTED_TEST_COLUMNS].isnull().any().any():
        raise ValueError("Test examples CSV contains blank text or label values.")

    invalid_labels = sorted(set(test_df["expected_label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Test examples CSV contains invalid labels: {invalid_labels}")

    return test_df


def build_agent_assessment_for_test(text, prediction, confidence):
    labels = ["borderline", "needs_more_info", "routine", "strong_sred"]
    probabilities = [0.1, 0.1, 0.1, 0.1]
    probabilities[labels.index(prediction)] = confidence

    signals = extract_signals(text)
    questions = generate_followup_questions(prediction, signals)
    cra_reference_questions = generate_cra_reference_questions()
    evidence_map = map_to_sred_framework(text, signals)
    cra_check = check_against_cra_guidelines(text, signals)
    explanation = generate_label_explanation(
        prediction,
        probabilities,
        labels,
        signals,
        cra_check,
    )

    return build_agent_assessment(
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


def validate_agent_assessment_layer():
    cases = [
        {
            "name": "strong candidate moves to evidence collection",
            "text": (
                "Known methods were insufficient for distorted OCR extraction. "
                "The team formed a hypothesis, tested multiple preprocessing approaches, "
                "measured benchmark results, learned which method reduced false positives, "
                "and recorded the work in Jira tickets and Git commits."
            ),
            "prediction": "strong_sred",
            "confidence": 0.62,
            "expected_stage": "evidence_collection",
            "expected_priority": "high",
        },
        {
            "name": "vague project moves to technical interview",
            "text": "We improved the backend system.",
            "prediction": "needs_more_info",
            "confidence": 0.58,
            "expected_stage": "technical_interview",
            "expected_priority": "medium",
        },
        {
            "name": "routine project stays in routine screening",
            "text": (
                "The team configured an off-the-shelf dashboard using vendor documentation "
                "and standard reporting tools."
            ),
            "prediction": "routine",
            "confidence": 0.61,
            "expected_stage": "routine_screening",
            "expected_priority": "low",
        },
    ]

    print("\nAgentic assessment layer checks")
    print("=" * 80)

    for case in cases:
        assessment = build_agent_assessment_for_test(
            case["text"],
            case["prediction"],
            case["confidence"],
        )

        if assessment["case_stage"] != case["expected_stage"]:
            raise AssertionError(
                f"{case['name']} expected stage {case['expected_stage']}, "
                f"found {assessment['case_stage']}"
            )

        if assessment["priority"] != case["expected_priority"]:
            raise AssertionError(
                f"{case['name']} expected priority {case['expected_priority']}, "
                f"found {assessment['priority']}"
            )

        required_lists = [
            "blockers",
            "evidence_requests",
            "interview_focus",
            "action_plan",
        ]
        for key in required_lists:
            if not assessment[key]:
                raise AssertionError(f"{case['name']} returned an empty {key} list.")

        if not assessment["handoff_summary"]:
            raise AssertionError(f"{case['name']} returned an empty handoff summary.")

        print(f"PASS: {case['name']}")


def validate_intake_agent_layer(model):
    initial_text = "We improved the backend system."
    initial_analysis = analyze_text(initial_text, model)
    questions = select_intake_questions(
        initial_analysis["agent_assessment"],
        max_questions=3,
    )

    if not questions:
        raise AssertionError("Intake agent did not generate follow-up questions.")

    if len(questions) > 3:
        raise AssertionError("Intake agent returned more questions than requested.")

    if len(questions) != len(set(questions)):
        raise AssertionError("Intake agent returned duplicate questions.")

    answers = [
        {
            "question": questions[0],
            "answer": (
                "Standard database indexing did not resolve write latency under "
                "concurrent transaction loads, so the team tested partitioning "
                "and queueing approaches."
            ),
        }
    ]
    updated_text = build_updated_description(initial_text, answers)

    required_fragments = [
        initial_text,
        "Follow-up intake answers:",
        f"Question: {questions[0]}",
        answers[0]["answer"],
    ]
    for fragment in required_fragments:
        if fragment not in updated_text:
            raise AssertionError(f"Updated intake description is missing: {fragment}")

    final_analysis = analyze_text(updated_text, model)
    required_keys = [
        "prediction",
        "agent_assessment",
        "explanation",
        "cra_check",
    ]
    for key in required_keys:
        if key not in final_analysis:
            raise AssertionError(f"Final intake analysis is missing: {key}")

    if not final_analysis["agent_assessment"]["action_plan"]:
        raise AssertionError("Final intake assessment returned an empty action plan.")

    session = {
        "initial_analysis": initial_analysis,
        "questions": questions,
        "answers": answers,
        "updated_text": updated_text,
        "final_analysis": final_analysis,
        "report_path": BASE_DIR / "reports" / "test_report.txt",
    }

    with TemporaryDirectory() as temp_dir:
        case_path = save_intake_case_file(session, Path(temp_dir))
        case_data = json.loads(case_path.read_text(encoding="utf-8"))

    expected_case_fields = [
        "case_id",
        "created_at",
        "original_text",
        "questions",
        "answers",
        "updated_text",
        "report_path",
        "initial_assessment",
        "final_assessment",
    ]
    for field in expected_case_fields:
        if field not in case_data:
            raise AssertionError(f"Case file is missing: {field}")

    if case_data["original_text"] != initial_text:
        raise AssertionError("Case file did not preserve the original text.")

    if case_data["answers"][0]["answer"] != answers[0]["answer"]:
        raise AssertionError("Case file did not preserve the intake answer.")

    if case_data["final_assessment"]["prediction"] != final_analysis["prediction"]:
        raise AssertionError("Case file final assessment prediction is incorrect.")

    print("\nGuided intake agent checks")
    print("=" * 80)
    print("PASS: generated prioritized intake questions")
    print("PASS: built updated case description from answers")
    print("PASS: re-analyzed updated case description")
    print("PASS: saved reusable JSON case file")


def build_sample_case_session(model):
    initial_text = "We improved the backend system."
    initial_analysis = analyze_text(initial_text, model)
    questions = select_intake_questions(
        initial_analysis["agent_assessment"],
        max_questions=3,
    )
    answers = [
        {
            "question": questions[0],
            "answer": (
                "Standard database indexing did not resolve write latency under "
                "concurrent transaction loads."
            ),
        },
        {
            "question": questions[1],
            "answer": (
                "The team tested partitioning, queueing, and batching approaches, "
                "then compared latency and error rates across iterations."
            ),
        },
    ]
    updated_text = build_updated_description(initial_text, answers)
    final_analysis = analyze_text(updated_text, model)

    return {
        "initial_analysis": initial_analysis,
        "questions": questions,
        "answers": answers,
        "updated_text": updated_text,
        "final_analysis": final_analysis,
        "report_path": BASE_DIR / "reports" / "sample_report.txt",
    }


def validate_case_manager_layer(model):
    session = build_sample_case_session(model)

    with TemporaryDirectory() as temp_dir:
        case_path = save_intake_case_file(session, Path(temp_dir))
        case_id = case_path.stem

        listed_cases = list_case_files(Path(temp_dir))
        if len(listed_cases) != 1:
            raise AssertionError("Case listing did not return the saved case.")

        printed_cases = list_cases(Path(temp_dir))
        if len(printed_cases) != 1:
            raise AssertionError("Case manager list command did not return the saved case.")

        loaded_case = load_case(case_id, Path(temp_dir))
        if loaded_case["case_id"] != case_id:
            raise AssertionError("Loaded case ID does not match the saved case.")

        inspected_case = inspect_case(case_id, Path(temp_dir))
        if inspected_case["case_id"] != case_id:
            raise AssertionError("Inspect command returned the wrong case.")

        listing_summary = summarize_case_for_listing(loaded_case)
        if not listing_summary["summary"]:
            raise AssertionError("Case listing summary is empty.")

        resumed_text = build_resumed_description(
            loaded_case["updated_text"],
            [
                {
                    "question": "What evidence supports the investigation?",
                    "answer": "Jira tickets and benchmark logs are available.",
                }
            ],
        )
        if "Additional follow-up intake answers:" not in resumed_text:
            raise AssertionError("Resume description did not add the resume answer section.")

        original_collect = case_manager.collect_intake_answers
        case_manager.collect_intake_answers = lambda questions: [
            {
                "question": questions[0],
                "answer": "The team can provide Jira tickets, commits, and benchmark logs.",
            }
        ]
        try:
            resume_result = resume_case(
                case_id,
                model,
                base_dir=Path(temp_dir),
                max_questions=1,
            )
        finally:
            case_manager.collect_intake_answers = original_collect

        resumed_case_path = resume_result["case_file_path"]
        if resumed_case_path is None:
            raise AssertionError("Resume command did not save a new case file.")

        resumed_case = json.loads(resumed_case_path.read_text(encoding="utf-8"))
        if resumed_case.get("parent_case_id") != case_id:
            raise AssertionError("Resumed case did not record its parent case ID.")

        if resumed_case["status"] != "resumed":
            raise AssertionError("Resumed case did not record resumed status.")

    print("\nCase manager checks")
    print("=" * 80)
    print("PASS: listed saved case files")
    print("PASS: inspected saved case file")
    print("PASS: resumed case file and saved child case")


def validate_technical_report_layer(model):
    session = build_sample_case_session(model)

    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        case_path = save_intake_case_file(session, base_dir)
        case_data = load_case(case_path.stem, base_dir)
        report = build_technical_report(case_data)
        readiness = assess_report_readiness(case_data["final_assessment"])
        report_path = generate_technical_report_for_case(case_path.stem, base_dir)
        report_text = report_path.read_text(encoding="utf-8")

    required_sections = [
        "Executive Summary",
        "Project Overview",
        "Technological Uncertainty",
        "Systematic Investigation",
        "Results And Technical Learning",
        "Supporting Evidence Inventory",
        "Open Gaps And Analyst Questions",
    ]
    rendered_headings = [section["heading"] for section in report["sections"]]
    for section in required_sections:
        if section not in rendered_headings:
            raise AssertionError(f"Technical report is missing section: {section}")

    required_fragments = [
        "# SR&ED Technical Report Draft",
        f"Case ID: {case_path.stem}",
        "Report readiness",
        "Standard database indexing did not resolve write latency",
        "Supporting Evidence Inventory",
    ]
    for fragment in required_fragments:
        if fragment not in report_text:
            raise AssertionError(f"Technical report text is missing: {fragment}")

    if readiness["score"] <= 0:
        raise AssertionError("Technical report readiness score was not calculated.")

    if readiness["level"] not in {
        "draft_ready",
        "draft_with_gaps",
        "intake_required",
        "not_report_ready",
    }:
        raise AssertionError("Technical report readiness level is invalid.")

    print("\nTechnical report generator checks")
    print("=" * 80)
    print("PASS: built structured technical report draft")
    print("PASS: saved Markdown technical report")
    print("PASS: calculated report readiness")


training_df = validate_training_csv()
print("\nTraining CSV integrity check")
print("=" * 80)
print("Status: PASS")
print(f"Rows: {len(training_df)}")
print("Label counts:")
print(training_df["label"].value_counts().sort_index().to_string())

model = joblib.load(MODEL_PATH)
df = validate_test_examples_csv()

correct = 0
total = len(df)

expected_counter = Counter()
predicted_counter = Counter()
mistakes = []

print("\nSR&ED Classifier Test Results")
print("=" * 80)

for index, row in df.iterrows():
    text = row["text"]
    expected_label = row["expected_label"]

    predicted_label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities)

    expected_counter[expected_label] += 1
    predicted_counter[predicted_label] += 1

    is_correct = predicted_label == expected_label

    if is_correct:
        correct += 1
        result = "PASS"
    else:
        result = "FAIL"
        mistakes.append({
            "test_number": index + 1,
            "expected": expected_label,
            "predicted": predicted_label,
            "confidence": confidence,
            "text": text
        })

    print("\n" + "-" * 80)
    print(f"Test {index + 1}: {result}")
    print(f"Expected: {expected_label}")
    print(f"Predicted: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")
    print("Text:")
    print(text)

accuracy = correct / total if total else 0

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Passed: {correct}/{total}")
print(f"Accuracy: {accuracy:.2%}")

print("\nExpected label counts:")
for label, count in expected_counter.items():
    print(f"- {label}: {count}")

print("\nPredicted label counts:")
for label, count in predicted_counter.items():
    print(f"- {label}: {count}")

if mistakes:
    print("\nMistakes to review:")
    for mistake in mistakes:
        print("\n" + "-" * 80)
        print(f"Test {mistake['test_number']}")
        print(f"Expected: {mistake['expected']}")
        print(f"Predicted: {mistake['predicted']}")
        print(f"Confidence: {mistake['confidence']:.2f}")
        print("Text:")
        print(mistake["text"])
else:
    print("\nNo mistakes found.")

validate_agent_assessment_layer()
validate_intake_agent_layer(model)
validate_case_manager_layer(model)
validate_technical_report_layer(model)

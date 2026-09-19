from pathlib import Path
from collections import Counter

import pandas as pd
import joblib

from agent_assessment import build_agent_assessment
from cra_guideline_checker import check_against_cra_guidelines
from evidence_mapper import map_to_sred_framework
from explanation import generate_label_explanation
from questions import generate_followup_questions, generate_cra_reference_questions
from rules import extract_signals


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

from pathlib import Path
from collections import Counter

import pandas as pd
import joblib


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

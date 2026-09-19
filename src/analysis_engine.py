from pathlib import Path

import joblib

from agent_assessment import build_agent_assessment
from cra_guideline_checker import check_against_cra_guidelines
from evidence_mapper import map_to_sred_framework
from explanation import generate_label_explanation
from questions import generate_followup_questions, generate_cra_reference_questions
from recommendation import generate_recommendation
from rules import extract_signals


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "sred_classifier.joblib"


def load_classifier(model_path=MODEL_PATH):
    return joblib.load(model_path)


def analyze_text(text, model):
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
        signals,
    )
    evidence_map = map_to_sred_framework(text, signals)
    cra_check = check_against_cra_guidelines(text, signals)
    explanation = generate_label_explanation(
        prediction,
        probabilities,
        labels,
        signals,
        cra_check,
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

    return {
        "text": text,
        "prediction": prediction,
        "probabilities": probabilities,
        "labels": labels,
        "signals": signals,
        "questions": questions,
        "cra_reference_questions": cra_reference_questions,
        "recommendation": recommendation,
        "evidence_map": evidence_map,
        "cra_check": cra_check,
        "explanation": explanation,
        "agent_assessment": agent_assessment,
    }

import json
from datetime import datetime
from pathlib import Path


CASE_FILES_DIR = "case_files"


def save_intake_case_file(session, base_dir, timestamp=None):
    case_dir = Path(base_dir) / CASE_FILES_DIR
    case_dir.mkdir(exist_ok=True)

    record = build_case_record(session, timestamp=timestamp)
    case_path = unique_case_path(case_dir, record["case_id"])
    record["case_id"] = case_path.stem

    case_path.write_text(
        json.dumps(make_json_safe(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return case_path


def build_case_record(session, timestamp=None):
    created_at = timestamp or datetime.now().astimezone()
    initial_analysis = session["initial_analysis"]
    final_analysis = session["final_analysis"]

    return {
        "case_id": created_at.strftime("case_%Y%m%d_%H%M%S"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "status": "intake_completed",
        "original_text": initial_analysis["text"],
        "questions": session["questions"],
        "answers": session["answers"],
        "updated_text": session["updated_text"],
        "report_path": str(session["report_path"]),
        "initial_assessment": summarize_analysis(initial_analysis),
        "final_assessment": summarize_analysis(final_analysis),
    }


def summarize_analysis(analysis):
    return {
        "prediction": analysis["prediction"],
        "category_probabilities": build_probability_map(
            analysis["labels"],
            analysis["probabilities"],
        ),
        "signals": analysis["signals"],
        "cra_alignment": analysis["cra_check"]["overall_alignment"],
        "cra_checks": analysis["cra_check"]["checks"],
        "explanation": analysis["explanation"],
        "agent_assessment": analysis["agent_assessment"],
        "recommendation": analysis["recommendation"],
        "evidence_map": analysis["evidence_map"],
    }


def build_probability_map(labels, probabilities):
    return {
        str(label): float(probability)
        for label, probability in zip(labels, probabilities)
    }


def unique_case_path(case_dir, case_id):
    case_path = case_dir / f"{case_id}.json"

    if not case_path.exists():
        return case_path

    suffix = 2
    while True:
        candidate_path = case_dir / f"{case_id}_{suffix}.json"
        if not candidate_path.exists():
            return candidate_path
        suffix += 1


def make_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())

    if hasattr(value, "item"):
        return value.item()

    return value

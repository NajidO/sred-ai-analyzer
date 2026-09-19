import json
from datetime import datetime
from pathlib import Path


CASE_FILES_DIR = "case_files"


def get_case_dir(base_dir):
    return Path(base_dir) / CASE_FILES_DIR


def save_intake_case_file(session, base_dir, timestamp=None):
    case_dir = get_case_dir(base_dir)
    case_dir.mkdir(exist_ok=True)

    record = build_case_record(session, timestamp=timestamp)
    case_path = unique_case_path(case_dir, record["case_id"])
    record["case_id"] = case_path.stem

    case_path.write_text(
        json.dumps(make_json_safe(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return case_path


def list_case_files(base_dir):
    case_dir = get_case_dir(base_dir)

    if not case_dir.exists():
        return []

    cases = [
        load_case_file(path)
        for path in sorted(case_dir.glob("*.json"))
    ]

    return sorted(
        cases,
        key=lambda case: case["created_at"],
        reverse=True,
    )


def resolve_case_path(case_reference, base_dir):
    case_path = Path(case_reference)

    if case_path.exists():
        return case_path

    case_dir = get_case_dir(base_dir)
    if case_reference.endswith(".json"):
        candidate_path = case_dir / case_reference
    else:
        candidate_path = case_dir / f"{case_reference}.json"

    if candidate_path.exists():
        return candidate_path

    raise FileNotFoundError(f"Could not find case file: {case_reference}")


def load_case_file(case_path):
    path = Path(case_path)
    case_data = json.loads(path.read_text(encoding="utf-8"))
    case_data["_path"] = str(path)
    return case_data


def load_case(case_reference, base_dir):
    return load_case_file(resolve_case_path(case_reference, base_dir))


def summarize_case_for_listing(case_data):
    final_assessment = case_data.get("final_assessment", {})
    agent_assessment = final_assessment.get("agent_assessment", {})

    return {
        "case_id": case_data.get("case_id", ""),
        "created_at": case_data.get("created_at", ""),
        "status": case_data.get("status", ""),
        "prediction": final_assessment.get("prediction", ""),
        "case_stage": agent_assessment.get("case_stage", ""),
        "priority": agent_assessment.get("priority", ""),
        "summary": build_text_preview(case_data.get("updated_text", "")),
        "path": case_data.get("_path", ""),
    }


def build_text_preview(text, max_length=90):
    compact_text = " ".join(text.split())

    if len(compact_text) <= max_length:
        return compact_text

    return compact_text[: max_length - 3] + "..."


def build_case_record(session, timestamp=None):
    created_at = timestamp or datetime.now().astimezone()
    initial_analysis = session["initial_analysis"]
    final_analysis = session["final_analysis"]
    status = session.get("status", "intake_completed")

    record = {
        "case_id": created_at.strftime("case_%Y%m%d_%H%M%S"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "status": status,
        "original_text": initial_analysis["text"],
        "questions": session["questions"],
        "answers": session["answers"],
        "updated_text": session["updated_text"],
        "report_path": str(session["report_path"]),
        "initial_assessment": summarize_analysis(initial_analysis),
        "final_assessment": summarize_analysis(final_analysis),
    }

    if session.get("parent_case_id"):
        record["parent_case_id"] = session["parent_case_id"]

    return record



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

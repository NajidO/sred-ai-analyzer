from pathlib import Path
from collections import Counter
from tempfile import TemporaryDirectory
import json

import pandas as pd
import joblib

import case_manager
from analysis_engine import analyze_text
from agent_assessment import build_agent_assessment
from capability_eval import evaluate_expected_gaps
from case_manager import build_resumed_description, inspect_case, list_cases, resume_case
from cra_guideline_checker import check_against_cra_guidelines
from evidence_mapper import map_to_sred_framework
from explanation import generate_label_explanation
from intake_agent import build_updated_description, select_intake_questions
from llm_report_agent import (
    build_local_context,
    build_model_input,
    find_new_measurements,
    normalize_report_payload,
    render_llm_report,
)
from case_store import (
    list_case_files,
    load_case,
    save_intake_case_file,
    summarize_case_for_listing,
)
from questions import generate_followup_questions, generate_cra_reference_questions
from rules import extract_signals
from report_strategy import build_report_strategy
from technical_report import (
    T661_LINE_WORD_LIMITS,
    assess_report_readiness,
    build_technical_report,
    build_t661_project_description,
    extract_questionnaire_sections,
    generate_technical_report_for_case,
    render_technical_report,
    word_count,
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
        "# SR&ED T661 Evidence Assessment",
        f"Case ID: {case_path.stem}",
        "T661 Evidence Assessment",
        "T661 Drafting Decision",
        "Draft not generated",
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

    if report["t661_evidence_assessment"]["can_draft"]:
        raise AssertionError("Incomplete sample case was incorrectly marked draft-ready.")

    if report["t661_project_description"]:
        raise AssertionError("Incomplete sample case produced a partial T661 draft.")

    if report["readiness"]["level"] != "intake_required":
        raise AssertionError("Evidence gate did not set incomplete case to intake_required.")

    print("\nTechnical report generator checks")
    print("=" * 80)
    print("PASS: built structured T661 evidence assessment")
    print("PASS: saved Markdown technical report")
    print("PASS: calculated report readiness")
    print("PASS: withheld T661 drafting when required evidence was missing")


def validate_t661_questionnaire_drafting(model):
    questionnaire_text = """
# SR&ED Technical Project Questionnaire

## 1. Please provide an overview of the project. What were you trying to develop or improve?

The objective was to develop a compact scanning electron microscopy platform with a shorter electron-optical column while maintaining approximately 10 nm imaging resolution and low image drift.

## 2. What was the technology you had available at the beginning of the project?

The company had an existing larger SEM design and understood standard focusing, stigmation, detector gain, vacuum control, and drift correction methods. These techniques worked in the larger architecture because there was more physical space for lens assemblies, shielding, and thermal isolation.

## 3. What specific technological challenge or uncertainty did you face?

The uncertainty was whether a beam with sufficient spot size, stability, and detector signal could be produced in the shortened column without unacceptable aberrations, thermal drift, magnetic hysteresis, and detector field interactions.

## 4. Why could this not simply be solved using standard engineering methods or generally available knowledge?

Published electron-optics references and supplier specifications did not provide a design that predicted the combined behaviour of the shortened integrated column. Established calculations could estimate individual parameters but did not accurately predict integrated beam stability after thermal, magnetic, mechanical, and detector effects interacted.

## 5. What was the initial technological idea or hypothesis?

The first hypothesis was that reducing pole-piece separation and increasing objective-lens excitation would compensate for the shorter electron-optical distance, while passive heat transfer would control the added temperature.

## 6. Describe the first approach tested during the fiscal year.

The team produced a shortened-column prototype and tested combinations of lens current, working distance, aperture diameter, and accelerating voltage. At 5 kV the prototype produced approximately 18-25 nm resolution. Higher objective current temporarily improved resolution to approximately 14-16 nm, but heating caused focus drift and repeatability problems. The scaled design was rejected.

## 7. What did you try next and why?

The team tested three pole-piece variants using electromagnetic simulations and prototype inserts. Variant C reduced objective-lens current by approximately 17% and achieved approximately 11-13 nm resolution, but drift remained approximately 25-35 nm/minute after ten minutes.

## 8. How did the results change your understanding or lead to the next experiment?

The team instrumented the objective assembly with temperature sensors and tested thermal mass and heat-spreading structures. Passive thermal changes improved drift to approximately 15-25 nm/minute but did not consistently meet the 10 nm/minute objective.

## 9. Was there another hypothesis or approach after that?

The team developed active compensation using temperature, commanded lens current, previous current state, and elapsed time. Combining model-based compensation with image-based drift estimation reduced steady-state apparent drift to approximately 6-9 nm/minute under reference conditions, but low-contrast samples remained problematic.

## 10. Were there any other technical problems investigated?

The team investigated secondary-electron detector geometry, detector bias, and shielding. A revised detector configuration improved SNR from approximately 6-8 to approximately 13-16 at comparable low beam current.

## 11. What testing or analysis did you perform to determine whether the approaches worked?

The team recorded accelerating voltage, beam current, lens current, working distance, aperture configuration, detector settings, temperature, image sequences, drift measurements, detector SNR, and simulation results.

## 12. Summarize the major experimental results.

Resolution improved from approximately 18-25 nm to approximately 11-13 nm. Drift under reference conditions improved from approximately 40-60 nm/minute to approximately 6-9 nm/minute. Detector SNR improved from approximately 6-8 to approximately 13-16, although not all conditions were resolved.

## 13. What technological knowledge did you gain during the fiscal year?

The team determined that scaling the electron-optical geometry and increasing objective-lens excitation was not viable because thermal and magnetic-history effects prevented stable operation. The team gained knowledge about pole-piece return path geometry, lens excitation history, thermal state, and detector electric-field interactions.

## 14. What measurable improvements resulted from the experimental work?

The later prototype achieved approximately 11-13 nm resolution, reduced objective-lens excitation by approximately 17%, reduced steady-state drift to approximately 6-9 nm/minute under reference conditions, and increased detector SNR to approximately 13-16.

## 15. Did any failed work generate useful technological knowledge?

The failed high-current, high-thermal-mass, linear temperature compensation, and first annular detector approaches showed which mechanisms were insufficient and changed the understanding of the compact architecture.

## 16. What remained technologically uncertain at the end of the fiscal year?

Drift compensation for low-contrast samples and performance across the complete voltage, working-distance, and sample-type envelope remained technologically uncertain.
"""
    analysis = analyze_text(questionnaire_text, model)
    session = {
        "initial_analysis": analysis,
        "questions": [],
        "answers": [],
        "updated_text": questionnaire_text,
        "final_analysis": analysis,
        "report_path": BASE_DIR / "reports" / "t661_test_report.txt",
    }

    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        case_path = save_intake_case_file(session, base_dir)
        case_data = load_case(case_path.stem, base_dir)
        source_sections = extract_questionnaire_sections(questionnaire_text)
        technical_report = build_technical_report(case_data)
        t661_sections = build_t661_project_description(
            case_data,
            case_data["final_assessment"],
            source_sections,
        )
        report_path = generate_technical_report_for_case(case_path.stem, base_dir)
        report_text = report_path.read_text(encoding="utf-8")

    if len(source_sections) < 16:
        raise AssertionError("Questionnaire parser did not capture the numbered sections.")

    if not technical_report["t661_evidence_assessment"]["can_draft"]:
        raise AssertionError("Complete questionnaire was incorrectly blocked from drafting.")

    for line_number, line in t661_sections.items():
        if not line["draft"]:
            raise AssertionError(f"T661 line {line_number} draft is empty.")

        if word_count(line["draft"]) > T661_LINE_WORD_LIMITS[line_number]:
            raise AssertionError(f"T661 line {line_number} exceeds its word limit.")

    if "18-25 nm" not in t661_sections["244"]["draft"]:
        raise AssertionError("Line 244 did not include experimental measurement detail.")

    if "thermal and magnetic-history effects" not in t661_sections["246"]["draft"]:
        raise AssertionError("Line 246 did not include technological learning detail.")

    if "TU1 -" not in t661_sections["242"]["draft"]:
        raise AssertionError("Line 242 did not include technical uncertainty labels.")

    if "SIS1 for TU1" not in t661_sections["244"]["draft"]:
        raise AssertionError("Line 244 did not include systematic investigation labels.")

    if "TU1/SIS1 advancement" not in t661_sections["246"]["draft"]:
        raise AssertionError("Line 246 did not link advancements to TU/SIS labels.")

    if "## T661 Project Description Draft" not in report_text:
        raise AssertionError("Rendered report is missing the T661 section.")

    print("\nT661 project description checks")
    print("=" * 80)
    print("PASS: parsed numbered questionnaire sections")
    print("PASS: drafted T661 lines 242, 244, and 246")
    print("PASS: allowed drafting only after all three lines passed evidence checks")
    print("PASS: kept T661 drafts within CRA word limits")
    print("PASS: sectioned T661 drafts with TU/SIS labels when useful")


def validate_report_strategy_layer(model):
    questionnaire_text = """
# SR&ED Technical Project Questionnaire

## 1. Please provide an overview of the project. What were you trying to develop or improve?

The objective was to develop a compact scanning electron microscopy platform with a shorter electron-optical column while maintaining 10 nm imaging resolution, low image drift, and acceptable detector SNR.

## 3. What specific technological challenge or uncertainty did you face?

The uncertainty involved objective lens geometry, pole-piece spacing, thermal drift, magnetic hysteresis, active compensation, detector bias, shielding, and secondary-electron signal-to-noise ratio in the compact chamber.

## 6. Describe the first approach tested during the fiscal year.

The team tested pole-piece geometry, objective-lens current, aperture size, working distance, and accelerating voltage. Some configurations improved resolution but failed due to drift and focus repeatability.

## 8. How did the results change your understanding or lead to the next experiment?

The team tested thermal mass, heat-spreading structures, temperature sensing, and drift measurements. Passive thermal management did not consistently meet the drift target.

## 9. Was there another hypothesis or approach after that?

The team tested active compensation using lens temperature, current history, elapsed time, focus correction, and image-based drift estimation.

## 10. Were there any other technical problems investigated?

The team tested detector geometry, detector distance, bias voltage, shielding, beam current, charging effects, and SNR measurements.
"""
    analysis = analyze_text(questionnaire_text, model)
    session = {
        "initial_analysis": analysis,
        "questions": [],
        "answers": [],
        "updated_text": questionnaire_text,
        "final_analysis": analysis,
        "report_path": BASE_DIR / "reports" / "strategy_test_report.txt",
    }

    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        case_path = save_intake_case_file(session, base_dir)
        case_data = load_case(case_path.stem, base_dir)
        source_sections = extract_questionnaire_sections(questionnaire_text)
        strategy = build_report_strategy(
            case_data,
            case_data["final_assessment"],
            source_sections,
        )
        report = build_technical_report(case_data)
        report_path = generate_technical_report_for_case(case_path.stem, base_dir)
        report_text = report_path.read_text(encoding="utf-8")

    if strategy["selected_structure"]["mode"] != "split_by_uncertainty_stream":
        raise AssertionError("Report strategy did not choose TU/SIS splitting.")

    if len(strategy["streams"]) < 3:
        raise AssertionError("Report strategy did not detect multiple technical streams.")

    expected_streams = {"TU1", "TU2", "TU3"}
    detected_streams = {
        stream["id"]
        for stream in strategy["streams"]
    }
    if not expected_streams.issubset(detected_streams):
        raise AssertionError("Report strategy did not detect the expected SEM streams.")

    if not any("pole-piece variants" in question for question in strategy["specific_questions"]):
        raise AssertionError("Report strategy did not generate stream-specific questions.")

    if "Drafting Strategy And Rationale" not in report_text:
        raise AssertionError("Rendered report is missing strategy rationale.")

    if "Candidate TU/SIS Streams" not in report_text:
        raise AssertionError("Rendered report is missing candidate streams.")

    if report["report_strategy"]["selected_structure"]["mode"] != "split_by_uncertainty_stream":
        raise AssertionError("Built report did not include the selected strategy.")

    print("\nReport strategy planner checks")
    print("=" * 80)
    print("PASS: selected TU/SIS split when multiple uncertainties were detected")
    print("PASS: generated strategy rationale")
    print("PASS: generated specific stream-level follow-up questions")


def validate_llm_report_agent_layer():
    source_text = (
        "The team targeted 10 nm resolution. Standard scaling was insufficient, "
        "so it tested revised lens geometry and recorded the results."
    )
    payload = {
        "overall_assessment": "The supplied facts describe a testable technical issue.",
        "eligibility_signal": "possible_candidate",
        "confidence": "medium",
        "structure_mode": "integrated_narrative",
        "structure_rationale": "One uncertainty stream is clearest for this short record.",
        "drafting_decision": "draft_ready",
        "section_assessments": [
            {
                "line_number": "242",
                "status": "ready",
                "supported_information": ["A 10 nm target and standard-practice limit are stated."],
                "missing_information": [],
            },
            {
                "line_number": "244",
                "status": "ready",
                "supported_information": ["Revised lens geometry was tested."],
                "missing_information": [],
            },
            {
                "line_number": "246",
                "status": "ready",
                "supported_information": ["The intended technical learning is stated."],
                "missing_information": [],
            },
        ],
        "technical_streams": [
            {
                "id": "TU1",
                "title": "Lens geometry and resolution",
                "uncertainty": "Whether revised geometry could achieve 10 nm resolution.",
                "standard_practice_gap": "Standard scaling was reported as insufficient.",
                "systematic_investigation": "The team tested revised lens geometry.",
                "advancement": "The result requires analyst confirmation.",
                "source_support": ["The source states a 10 nm target."],
                "evidence_gaps": ["Provide the recorded test results."],
            }
        ],
        "line_242": "The team did not know whether revised geometry could achieve 10 nm resolution.",
        "line_244": "The team tested revised lens geometry and recorded the results.",
        "line_246": "The work sought knowledge about the effect of lens geometry on resolution.",
        "follow_up_questions": [
            {
                "question": "Which lens geometries were compared?",
                "why_it_matters": "The alternatives establish systematic investigation.",
                "examples_to_check": ["Design files", "Test matrix", "Image results"],
            }
        ],
        "factual_risks": [],
        "review_notes": ["Confirm every statement with project records."],
    }
    context = {
        "project_source": source_text,
        "local_analysis": {"prediction": "borderline"},
        "local_strategy": {
            "selected_structure": {"mode": "chronological_integrated"},
            "rationale": ["The local planner detected one stream."],
        },
        "local_t661_baseline": {},
    }

    model_input = build_model_input(context)
    if source_text not in model_input or "LOCAL ANALYZER CONTEXT" not in model_input:
        raise AssertionError("LLM model input omitted required context.")

    report = normalize_report_payload(payload, source_text)
    for line_number, line in report["t661_lines"].items():
        if word_count(line["draft"]) > T661_LINE_WORD_LIMITS[line_number]:
            raise AssertionError(f"AI Line {line_number} exceeds its word limit.")

    new_measurements = find_new_measurements(source_text, "The later test used 12 kV.")
    if "12 kv" not in new_measurements:
        raise AssertionError("Unsupported measurement detection missed a new value.")

    report_text = render_llm_report(
        report,
        context,
        "test-model",
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    required_fragments = [
        "AI-Assisted SR&ED Capability Test",
        "Line 242",
        "TU1 - Lens geometry and resolution",
        "Which lens geometries were compared?",
        "Input tokens: 100",
    ]
    for fragment in required_fragments:
        if fragment not in report_text:
            raise AssertionError(f"Rendered AI report is missing: {fragment}")

    incomplete_payload = dict(payload)
    incomplete_payload.update({
        "drafting_decision": "needs_more_information",
        "line_242": "",
        "line_244": "",
        "line_246": "",
        "section_assessments": [
            {
                "line_number": "242",
                "status": "needs_more_information",
                "supported_information": ["A lens geometry target is mentioned."],
                "missing_information": ["Clarify the exact technological uncertainty."],
            },
            {
                "line_number": "244",
                "status": "needs_more_information",
                "supported_information": ["A revised geometry was tested."],
                "missing_information": ["Provide alternatives, measurements, and decisions."],
            },
            {
                "line_number": "246",
                "status": "needs_more_information",
                "supported_information": [],
                "missing_information": ["State the technological knowledge gained."],
            },
        ],
    })
    incomplete_report = normalize_report_payload(incomplete_payload, source_text)
    if incomplete_report["t661_lines"]:
        raise AssertionError("Incomplete AI assessment produced partial T661 lines.")

    incomplete_text = render_llm_report(incomplete_report, context, "test-model")
    if "Draft not generated" not in incomplete_text:
        raise AssertionError("Incomplete AI report did not show the drafting block.")

    local_gate = {
        "sections": {
            line_number: {
                "status": "needs_more_information",
                "supported_information": [],
                "missing_information": [f"Missing evidence for Line {line_number}."],
            }
            for line_number in ("242", "244", "246")
        }
    }
    locally_blocked = normalize_report_payload(
        payload,
        source_text,
        drafting_allowed=False,
        local_evidence_assessment=local_gate,
    )
    if locally_blocked["drafting_decision"] != "needs_more_information":
        raise AssertionError("Local evidence gate did not override AI drafting.")
    if locally_blocked["t661_lines"]:
        raise AssertionError("Local evidence gate allowed AI-generated T661 prose.")

    print("\nLLM report agent checks")
    print("=" * 80)
    print("PASS: built grounded model input")
    print("PASS: validated structured report output")
    print("PASS: enforced T661 word limits locally")
    print("PASS: flagged unsupported measurements")
    print("PASS: rendered analyst-review Markdown")
    print("PASS: withheld AI-generated T661 prose when evidence was incomplete")
    print("PASS: enforced the local evidence gate over an AI drafting attempt")


def validate_incomplete_intake_capability(model):
    source_path = BASE_DIR / "examples" / "incomplete_sem_client_draft.md"
    expected_path = BASE_DIR / "examples" / "incomplete_sem_expected_gaps.json"
    source_text = source_path.read_text(encoding="utf-8")
    expected_gaps = json.loads(expected_path.read_text(encoding="utf-8"))
    context = build_local_context(source_text, classifier=model)
    evaluation = evaluate_expected_gaps(context, expected_gaps)
    strategy = context["local_strategy"]
    evidence_assessment = context["t661_evidence_assessment"]
    t661 = context["local_t661_baseline"]

    if evaluation["detected"] != evaluation["total"]:
        missed = [
            result["id"]
            for result in evaluation["results"]
            if not result["detected"]
        ]
        raise AssertionError(f"Incomplete intake evaluation missed gaps: {missed}")

    if strategy["selected_structure"]["mode"] != "split_by_uncertainty_stream":
        raise AssertionError("Incomplete intake did not select a TU/SIS structure.")

    detected_streams = {stream["id"] for stream in strategy["streams"]}
    if not {"TU1", "TU2", "TU3"}.issubset(detected_streams):
        raise AssertionError("Incomplete intake did not detect all three SEM streams.")

    if evidence_assessment["decision"] != "needs_more_information":
        raise AssertionError("Incomplete intake was incorrectly marked draft-ready.")

    if evidence_assessment["can_draft"]:
        raise AssertionError("Incomplete intake incorrectly allowed T661 drafting.")

    if t661:
        raise AssertionError("Incomplete intake produced a partial T661 report.")

    if set(evidence_assessment["blocked_lines"]) != {"242", "244", "246"}:
        raise AssertionError("Incomplete intake did not block all incomplete T661 lines.")

    line_244_questions = " ".join(
        question
        for stream in evidence_assessment["sections"]["244"]["stream_assessments"]
        for question in stream["questions"]
    ).lower()
    for required_question_detail in (
        "considered but not tested",
        "refined, abandoned, or replaced by a pivot",
        "what further",
        "what was actually measured",
    ):
        if required_question_detail not in line_244_questions:
            raise AssertionError(
                "Line 244 questions omitted required detail: "
                f"{required_question_detail}"
            )

    case_data = {
        "case_id": "incomplete_sem_test",
        "created_at": "",
        "status": "intake",
        "updated_text": source_text,
        "final_assessment": context["local_analysis"],
    }
    report_text = render_technical_report(build_technical_report(case_data))
    if "## T661 Project Description Draft" in report_text:
        raise AssertionError("Rendered incomplete assessment included a T661 draft.")
    if "**Draft not generated.**" not in report_text:
        raise AssertionError("Rendered incomplete assessment omitted the drafting decision.")

    unstructured_text = "\n".join(
        line
        for line in source_text.splitlines()
        if not line.startswith("#")
    )
    unstructured_context = build_local_context(unstructured_text, classifier=model)
    unstructured_assessment = unstructured_context["t661_evidence_assessment"]
    if unstructured_assessment["can_draft"]:
        raise AssertionError("Incomplete unstructured intake was incorrectly draft-ready.")
    if not unstructured_assessment["sections"]["244"]["supported_information"]:
        raise AssertionError("Unstructured intake work facts were not assessed.")

    blockers = " ".join(context["local_analysis"]["agent_assessment"]["blockers"])
    for expected_blocker in (
        "technological_advancement",
        "experimental_results",
        "supporting_evidence",
    ):
        if expected_blocker not in blockers:
            raise AssertionError(
                f"Incomplete intake did not flag blocker: {expected_blocker}"
            )

    print("\nIncomplete intake capability checks")
    print("=" * 80)
    print(
        f"PASS: detected {evaluation['detected']}/{evaluation['total']} "
        "deliberately omitted evidence categories"
    )
    print("PASS: selected and preserved TU1/TU2/TU3 structure")
    print("PASS: generated grounded stream-specific follow-up questions")
    print("PASS: blocked Lines 242, 244, and 246 when evidence was incomplete")
    print("PASS: generated no partial T661 report")
    print("PASS: assessed incomplete unstructured client narratives")


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
validate_t661_questionnaire_drafting(model)
validate_report_strategy_layer(model)
validate_llm_report_agent_layer()
validate_incomplete_intake_capability(model)

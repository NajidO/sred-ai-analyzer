from pathlib import Path
from collections import Counter
from tempfile import TemporaryDirectory
from io import BytesIO
import json
from urllib.error import HTTPError

import pandas as pd
import joblib

import case_manager
import run_live_agent_benchmark as live_agent_benchmark
from analysis_engine import analyze_text
from agent_assessment import build_agent_assessment
from capability_eval import evaluate_expected_gaps
from case_manager import build_resumed_description, inspect_case, list_cases, resume_case
from cra_guideline_checker import check_against_cra_guidelines
from evidence_mapper import map_to_sred_framework
from evidence_agent import (
    apply_evidence_audit,
    assess_evidence_graph,
    build_evidence_structure_plan,
    normalize_evidence_graph,
    validate_evidence_audit_payload,
)
from explanation import generate_label_explanation
from grounding import find_source_quote_locations, format_source_location
from intake_agent import build_updated_description, select_intake_questions
from llm_report_agent import (
    build_evidence_agent_report,
    build_local_context,
    build_model_input,
    build_responses_client,
    find_new_measurements,
    normalize_grounded_draft,
    normalize_report_payload,
    render_llm_report,
    request_llm_report,
    split_draft_claims,
    validate_draft_structure,
    validate_line_support,
    validate_structured_claim_support,
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
from responses_http_client import ResponsesHTTPClient
from run_capability_benchmark import run_benchmark
from run_semantic_evidence_benchmark import (
    run_benchmark as run_semantic_evidence_benchmark,
)
from run_live_agent_benchmark import (
    case_text as live_benchmark_case_text,
    evaluate_report as evaluate_live_agent_report,
    validate_benchmark_data as validate_live_benchmark_data,
)
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
BENCHMARK_PATHS = [
    BASE_DIR / "benchmarks" / "t661_capability_benchmark.json",
    BASE_DIR / "benchmarks" / "t661_holdout_benchmark.json",
    BASE_DIR / "benchmarks" / "t661_adversarial_benchmark.json",
]
SEMANTIC_EVIDENCE_BENCHMARK_PATH = (
    BASE_DIR / "benchmarks" / "semantic_evidence_gate_benchmark.json"
)
LIVE_AGENT_BENCHMARK_PATH = BASE_DIR / "benchmarks" / "live_agent_benchmark.json"


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


def build_semantic_evidence_fixture():
    source_text = "\n".join([
        "The team sought a controller that held vibration below 2 mm/s.",
        "At the start, the existing fixed-gain controller reached 8 mm/s.",
        "Published gain-tuning methods assumed a constant load and could not predict the changing-load response.",
        "It was unknown whether a load-state observer could preserve stability as the load changed.",
        "The team hypothesized that scheduling observer bandwidth from load state would reduce vibration without destabilizing the loop.",
        "During 2025, engineers tested four bandwidth schedules under three changing-load profiles.",
        "The selected schedule reached 1.8 mm/s, while the fastest schedule became unstable.",
        "The team concluded that observer bandwidth had to decrease as estimated load inertia increased.",
        "The work established the relationship between estimated load inertia, observer bandwidth, and closed-loop stability.",
        "Dated test logs, controller versions, vibration traces, and decision notes record the investigation.",
    ])
    evidence = [
        ("o1", "objective", "The team sought a controller that held vibration below 2 mm/s.", "The objective was to hold vibration below 2 mm/s."),
        ("k1", "existing_knowledge", "At the start, the existing fixed-gain controller reached 8 mm/s.", "The existing fixed-gain controller reached 8 mm/s."),
        ("sp1", "standard_practice_limit", "Published gain-tuning methods assumed a constant load and could not predict the changing-load response.", "Published gain-tuning methods could not predict the changing-load response."),
        ("u1", "uncertainty", "It was unknown whether a load-state observer could preserve stability as the load changed.", "It was unknown whether a load-state observer could preserve stability under changing load."),
        ("h1", "hypothesis", "The team hypothesized that scheduling observer bandwidth from load state would reduce vibration without destabilizing the loop.", "The hypothesis linked load-state bandwidth scheduling to vibration and stability."),
        ("x1", "experiment_or_analysis", "During 2025, engineers tested four bandwidth schedules under three changing-load profiles.", "Engineers tested four schedules under three profiles during 2025."),
        ("r1", "result", "The selected schedule reached 1.8 mm/s, while the fastest schedule became unstable.", "The selected schedule reached 1.8 mm/s and the fastest was unstable."),
        ("c1", "conclusion", "The team concluded that observer bandwidth had to decrease as estimated load inertia increased.", "The team concluded that bandwidth had to decrease as estimated inertia increased."),
        ("a1", "advancement", "The work established the relationship between estimated load inertia, observer bandwidth, and closed-loop stability.", "The work established a relationship among load inertia, observer bandwidth, and stability."),
        ("d1", "supporting_record", "Dated test logs, controller versions, vibration traces, and decision notes record the investigation.", "Dated technical records support the investigation."),
    ]
    payload = {
        "project_summary": "A changing-load vibration-control investigation.",
        "claimed_tax_year": "2025",
        "claimed_tax_year_source_quote": (
            "During 2025, engineers tested four bandwidth schedules under three changing-load profiles."
        ),
        "technical_streams": [
            {
                "id": "control_stream",
                "title": "Changing-load observer stability",
                "objective": "Hold vibration below 2 mm/s as load changes.",
                "relationship_to_other_streams": "Single technical stream.",
            }
        ],
        "evidence_items": [
            {
                "id": item_id,
                "stream_id": "control_stream",
                "category": category,
                "source_quote": quote,
                "source_location": f"fixture:{index}",
                "normalized_fact": fact,
                "certainty": "explicit",
                "tax_year_scope": "claimed_year",
                "attribution": "claimant",
            }
            for index, (item_id, category, quote, fact) in enumerate(evidence, start=1)
        ],
        "investigation_sequences": [
            {
                "id": "observer_sis",
                "stream_id": "control_stream",
                "title": "Observer bandwidth scheduling",
                "uncertainty_evidence_ids": ["u1"],
                "hypothesis_evidence_ids": ["h1"],
                "experiment_evidence_ids": ["x1"],
                "result_evidence_ids": ["r1"],
                "conclusion_evidence_ids": ["c1"],
                "advancement_evidence_ids": ["a1"],
            }
        ],
        "contradictions": [],
        "routine_work": [],
        "attribution_issues": [],
        "extraction_notes": [],
    }
    return source_text, payload


def build_grounded_draft_fixture():
    return {
        "overall_assessment": "The validated record supports a technical uncertainty, systematic investigation, and attempted advancement.",
        "structure_mode": "integrated_narrative",
        "structure_rationale": "One hypothesis and investigation address one uncertainty.",
        "line_242": "The team sought to control vibration under changing load. Fixed-gain control and published constant-load tuning did not predict the response, leaving uncertainty about whether a load-state observer could preserve stability.",
        "line_244": "The team hypothesized that scheduling observer bandwidth from load state would reduce vibration without destabilizing the loop. During 2025, engineers tested four schedules under three changing-load profiles. The selected schedule reached 1.8 mm/s, while the fastest schedule became unstable. The team concluded that bandwidth had to decrease as estimated load inertia increased.",
        "line_246": "The work established the relationship between estimated load inertia, observer bandwidth, and closed-loop stability.",
        "draft_support": [
            {
                "line_number": "242",
                "evidence_ids": ["E1", "E2", "E3", "E4"],
                "coverage_note": "Covers the objective, starting knowledge, standard-practice limit, and uncertainty.",
            },
            {
                "line_number": "244",
                "evidence_ids": ["E5", "E6", "E7", "E8", "E10"],
                "coverage_note": "Covers the hypothesis, work, result, conclusion, and records.",
            },
            {
                "line_number": "246",
                "evidence_ids": ["E9"],
                "coverage_note": "Covers the technological knowledge established.",
            },
        ],
        "claim_support": [
            {
                "claim_id": "C242_1",
                "line_number": "242",
                "claim_text": "The team sought to control vibration under changing load.",
                "evidence_ids": ["E1"],
            },
            {
                "claim_id": "C242_2",
                "line_number": "242",
                "claim_text": "Fixed-gain control and published constant-load tuning did not predict the response, leaving uncertainty about whether a load-state observer could preserve stability.",
                "evidence_ids": ["E2", "E3", "E4"],
            },
            {
                "claim_id": "C244_1",
                "line_number": "244",
                "claim_text": "The team hypothesized that scheduling observer bandwidth from load state would reduce vibration without destabilizing the loop.",
                "evidence_ids": ["E5"],
            },
            {
                "claim_id": "C244_2",
                "line_number": "244",
                "claim_text": "During 2025, engineers tested four schedules under three changing-load profiles.",
                "evidence_ids": ["E6"],
            },
            {
                "claim_id": "C244_3",
                "line_number": "244",
                "claim_text": "The selected schedule reached 1.8 mm/s, while the fastest schedule became unstable.",
                "evidence_ids": ["E7"],
            },
            {
                "claim_id": "C244_4",
                "line_number": "244",
                "claim_text": "The team concluded that bandwidth had to decrease as estimated load inertia increased.",
                "evidence_ids": ["E8"],
            },
            {
                "claim_id": "C246_1",
                "line_number": "246",
                "claim_text": "The work established the relationship between estimated load inertia, observer bandwidth, and closed-loop stability.",
                "evidence_ids": ["E9"],
            },
        ],
        "factual_risks": [],
        "review_notes": ["Verify technical terminology with the project lead."],
    }


def build_evidence_audit_fixture(
    graph,
    rejected_evidence_id=None,
    rejected_dimension="category",
    rejected_sequence_id=None,
    rejected_sequence_dimension="relationship",
    rejected_issue_id=None,
    tax_year_verdict=None,
    contradictions=None,
):
    dimensions = (
        "normalized_fact",
        "category",
        "certainty",
        "tax_year_scope",
        "attribution",
        "stream_assignment",
    )
    item_audits = []
    for item in graph["evidence_items"]:
        audit = {
            "evidence_id": item["id"],
            **{dimension: "supported" for dimension in dimensions},
            "reason": "The source supports every audited evidence dimension.",
        }
        if item["id"] == rejected_evidence_id:
            audit[rejected_dimension] = "unsupported"
            audit["reason"] = "The source does not support this semantic classification."
        item_audits.append(audit)
    sequence_audits = []
    for sequence in graph["investigation_sequences"]:
        audit = {
            "sequence_id": sequence["id"],
            "relationship": "supported",
            "chronology": "supported",
            "reason": "The source supports the sequence relationship and chronology.",
        }
        if sequence["id"] == rejected_sequence_id:
            audit[rejected_sequence_dimension] = "ambiguous"
            audit["reason"] = "The source does not establish this sequence linkage."
        sequence_audits.append(audit)
    if tax_year_verdict is None:
        tax_year_verdict = "supported" if graph["claimed_tax_year"] else "unsupported"
    issue_audits = []
    for issue in [*graph["routine_work"], *graph["attribution_issues"]]:
        rejected = issue["id"] == rejected_issue_id
        issue_audits.append({
            "issue_id": issue["id"],
            "verdict": "unsupported" if rejected else "supported",
            "reason": (
                "The quote does not support the extracted blocker interpretation."
                if rejected
                else "The source supports the extracted blocker interpretation."
            ),
        })
    return {
        "overall_assessment": (
            "One evidence item has an unsupported semantic classification."
            if rejected_evidence_id
            else "Every evidence item is semantically supported by the source."
        ),
        "claimed_tax_year_audit": {
            "verdict": tax_year_verdict,
            "reason": (
                "The quoted source establishes the claimed tax year."
                if tax_year_verdict == "supported"
                else "The source does not establish a unique claimed tax year."
            ),
        },
        "item_audits": item_audits,
        "sequence_audits": sequence_audits,
        "issue_audits": issue_audits,
        "discovered_contradictions": contradictions or [],
    }


def build_grounding_audit_fixture(draft_payload, rejected_claim_id=None):
    claims = []
    for claim in draft_payload["claim_support"]:
        rejected = claim["claim_id"] == rejected_claim_id
        claims.append({
            "claim_id": claim["claim_id"],
            "verdict": "unsupported" if rejected else "supported",
            "reason": (
                "The cited evidence does not establish every detail."
                if rejected
                else "The cited evidence supports the complete claim."
            ),
            "evidence_ids_reviewed": list(claim["evidence_ids"]),
        })
    return {
        "overall_assessment": (
            "One claim is unsupported."
            if rejected_claim_id
            else "Every drafted claim is supported by its cited evidence."
        ),
        "claims": claims,
    }


def validate_semantic_evidence_agent_layer():
    source_text, raw_graph = build_semantic_evidence_fixture()
    graph = normalize_evidence_graph(raw_graph, source_text)
    readiness = assess_evidence_graph(graph)

    if graph["validation"]["accepted_evidence_items"] != 10:
        raise AssertionError("The evidence validator rejected a supported fixture item.")
    if not readiness["can_draft"]:
        raise AssertionError("A complete validated evidence graph was not draft-ready.")

    topology_graph = {
        "technical_streams": [
            {"id": "TU1", "title": "Stream one"},
            {"id": "TU2", "title": "Stream two"},
        ],
        "investigation_sequences": [
            {"id": "SIS1", "stream_id": "TU1"},
            {"id": "SIS2", "stream_id": "TU2"},
        ],
        "evidence_items": [],
    }
    split_plan = build_evidence_structure_plan(topology_graph)
    if split_plan["mode"] != "split_by_uncertainty_stream":
        raise AssertionError("Distinct TU/SIS streams did not select split structure.")
    if split_plan["required_labels_by_line"] != {
        "242": ["TU1", "TU2"],
        "244": ["SIS1", "SIS2"],
        "246": ["TU1", "TU2"],
    }:
        raise AssertionError("The split structure plan returned the wrong line labels.")

    shared_topology = json.loads(json.dumps(topology_graph))
    shared_topology["evidence_items"] = [{
        "id": "E_SHARED",
        "stream_id": "GLOBAL",
        "category": "existing_knowledge",
    }]
    if build_evidence_structure_plan(shared_topology)["mode"] != "hybrid":
        raise AssertionError("Shared multi-stream context did not select hybrid structure.")

    multi_sequence_topology = {
        "technical_streams": [{"id": "TU1", "title": "Stream one"}],
        "investigation_sequences": [
            {"id": "SIS1", "stream_id": "TU1"},
            {"id": "SIS2", "stream_id": "TU1"},
        ],
        "evidence_items": [],
    }
    multi_sequence_plan = build_evidence_structure_plan(multi_sequence_topology)
    if multi_sequence_plan["mode"] != "hybrid":
        raise AssertionError("Multiple SIS chains in one TU did not select hybrid structure.")

    structured_payload = {
        "structure_mode": "split_by_uncertainty_stream",
        "structure_rationale": "Separate validated streams require separate sections.",
        "line_242": "### TU1\nFirst uncertainty.\n### TU2\nSecond uncertainty.",
        "line_244": "### SIS1\nFirst investigation.\n### SIS2\nSecond investigation.",
        "line_246": "### TU1\nFirst advancement.\n### TU2\nSecond advancement.",
    }
    validate_draft_structure(structured_payload, split_plan)
    heading_claims = split_draft_claims(
        "### SIS1 - Observer scheduling\n"
        "The team tested the first schedule.\n"
        "TU1: This prefixed sentence remains a factual claim."
    )
    if heading_claims != [
        "The team tested the first schedule.",
        "TU1: This prefixed sentence remains a factual claim.",
    ]:
        raise AssertionError("Structural headings were not separated from factual claims.")

    missing_structure_label = json.loads(json.dumps(structured_payload))
    missing_structure_label["line_244"] = "### SIS1\nFirst investigation."
    try:
        validate_draft_structure(missing_structure_label, split_plan)
    except ValueError as exc:
        if "omitted required Markdown structure headings: SIS2" not in str(exc):
            raise
    else:
        raise AssertionError("A multi-stream draft omitted a required SIS section.")

    empty_structure_section = json.loads(json.dumps(structured_payload))
    empty_structure_section["line_244"] = (
        "### SIS1\n\n### SIS2\nSecond investigation."
    )
    try:
        validate_draft_structure(empty_structure_section, split_plan)
    except ValueError as exc:
        if "SIS1 contained no substantive section content" not in str(exc):
            raise
    else:
        raise AssertionError("A required SIS heading contained no section content.")

    reversed_structure = json.loads(json.dumps(structured_payload))
    reversed_structure["line_242"] = (
        "### TU2\nSecond uncertainty.\n### TU1\nFirst uncertainty."
    )
    try:
        validate_draft_structure(reversed_structure, split_plan)
    except ValueError as exc:
        if "did not preserve the required TU/SIS order" not in str(exc):
            raise
    else:
        raise AssertionError("A draft reversed the evidence-driven stream order.")

    wrong_structure_mode = json.loads(json.dumps(structured_payload))
    wrong_structure_mode["structure_mode"] = "integrated_narrative"
    try:
        validate_draft_structure(wrong_structure_mode, split_plan)
    except ValueError as exc:
        if "ignored the evidence-driven structure mode" not in str(exc):
            raise
    else:
        raise AssertionError("A draft selected a structure that contradicted its graph.")

    structured_evidence = []
    structured_sequences = []
    for sequence_number, stream_id in ((1, "TU1"), (2, "TU2")):
        field_ids = {}
        for category, field_prefix in (
            ("hypothesis", "H"),
            ("experiment_or_analysis", "X"),
            ("result", "R"),
            ("conclusion", "C"),
        ):
            evidence_id = f"E_{field_prefix}{sequence_number}"
            field_ids[category] = evidence_id
            structured_evidence.append({
                "id": evidence_id,
                "stream_id": stream_id,
                "category": category,
                "certainty": "explicit",
                "tax_year_scope": "claimed_year",
                "attribution": "claimant",
            })
        structured_sequences.append({
            "id": f"SIS{sequence_number}",
            "stream_id": stream_id,
            "hypothesis_evidence_ids": [field_ids["hypothesis"]],
            "experiment_evidence_ids": [field_ids["experiment_or_analysis"]],
            "result_evidence_ids": [field_ids["result"]],
            "conclusion_evidence_ids": [field_ids["conclusion"]],
            "advancement_evidence_ids": [],
        })
    structured_claim_payload = {
        "line_244": "### SIS1\nFirst chain.\n### SIS2\nSecond chain.",
        "claim_support": [
            {
                "claim_id": "SC1",
                "line_number": "244",
                "claim_text": "First chain.",
                "evidence_ids": ["E_H1", "E_X1", "E_R1", "E_C1"],
            },
            {
                "claim_id": "SC2",
                "line_number": "244",
                "claim_text": "Second chain.",
                "evidence_ids": ["E_H2", "E_X2", "E_R2", "E_C2"],
            },
        ],
    }
    line_244_only_plan = json.loads(json.dumps(split_plan))
    line_244_only_plan["required_labels_by_line"]["242"] = []
    line_244_only_plan["required_labels_by_line"]["246"] = []
    structured_graph = {"investigation_sequences": structured_sequences}
    structured_evidence_index = {
        item["id"]: item for item in structured_evidence
    }
    structured_support = {
        "244": {"evidence_ids": list(structured_evidence_index)}
    }
    validate_structured_claim_support(
        structured_claim_payload,
        line_244_only_plan,
        structured_graph,
        structured_evidence_index,
        structured_support,
    )
    cross_labeled_claims = json.loads(json.dumps(structured_claim_payload))
    cross_labeled_claims["claim_support"][0]["evidence_ids"] = [
        "E_H2",
        "E_X2",
        "E_R2",
        "E_C2",
    ]
    try:
        validate_structured_claim_support(
            cross_labeled_claims,
            line_244_only_plan,
            structured_graph,
            structured_evidence_index,
            structured_support,
        )
    except ValueError as exc:
        if "Line 244 section SIS1 omitted its sequence evidence" not in str(exc):
            raise
    else:
        raise AssertionError("A SIS heading contained another sequence's grounded claims.")

    if graph["evidence_items"][0]["source_location"].startswith("fixture:"):
        raise AssertionError("A model-authored source location was trusted.")
    if not graph["evidence_items"][0]["source_location"].startswith(
        "line 1, column 1, characters 0-"
    ):
        raise AssertionError("The exact quote did not receive deterministic provenance.")

    typography_source = "Heading\nThe \u201cteam\u201d tested A \u2014 B.\nConclusion"
    typography_quote = 'The "team" tested A - B.'
    typography_locations = find_source_quote_locations(
        typography_source,
        typography_quote,
    )
    if len(typography_locations) != 1:
        raise AssertionError("Normalized typography did not retain source provenance.")
    if format_source_location(typography_locations[0]) != (
        "line 2, column 1, characters 8-32"
    ):
        raise AssertionError("Normalized quote provenance returned the wrong location.")

    multiline_locations = find_source_quote_locations(
        "Alpha beta\n gamma delta",
        "beta gamma",
    )
    if len(multiline_locations) != 1 or format_source_location(
        multiline_locations[0]
    ) != "lines 1-2, column 7, characters 6-17":
        raise AssertionError("Multi-line quote provenance returned the wrong location.")

    duplicate_source = source_text + "\n" + source_text.splitlines()[0]
    duplicate_graph = normalize_evidence_graph(raw_graph, duplicate_source)
    duplicate_rejections = duplicate_graph["validation"]["rejected_items"]
    if not any(
        "matched 2 locations" in reason
        for item in duplicate_rejections
        for reason in item["reasons"]
    ):
        raise AssertionError("An ambiguous repeated source quote was accepted.")
    duplicate_readiness = assess_evidence_graph(duplicate_graph)
    if duplicate_readiness["sections"]["242"]["status"] != "needs_more_information":
        raise AssertionError("Ambiguous objective provenance did not block Line 242.")

    duplicated_year_quote = raw_graph["claimed_tax_year_source_quote"]
    duplicate_year_graph = normalize_evidence_graph(
        raw_graph,
        source_text + "\n" + duplicated_year_quote,
    )
    if duplicate_year_graph["claimed_tax_year"]:
        raise AssertionError("An ambiguous claimed tax year source quote was accepted.")
    if not any(
        "Claimed tax year was cleared" in note
        for note in duplicate_year_graph["extraction_notes"]
    ):
        raise AssertionError("A rejected claimed tax year was not explained.")

    missing_year_payload = json.loads(json.dumps(raw_graph))
    missing_year_payload["claimed_tax_year"] = ""
    missing_year_payload["claimed_tax_year_source_quote"] = ""
    missing_year_graph = normalize_evidence_graph(missing_year_payload, source_text)
    missing_year_readiness = assess_evidence_graph(missing_year_graph)
    if missing_year_readiness["can_draft"] or set(
        missing_year_readiness["blocked_lines"]
    ) != {"242", "244", "246"}:
        raise AssertionError("An absent claimed tax year did not block every T661 line.")
    if not any(
        item["category"] == "claimed_tax_year"
        for item in missing_year_readiness["follow_up_questions"]
    ):
        raise AssertionError("A missing claimed tax year produced no targeted question.")

    mismatched_year_payload = json.loads(json.dumps(raw_graph))
    mismatched_year_payload["claimed_tax_year"] = "2026"
    mismatched_year_graph = normalize_evidence_graph(
        mismatched_year_payload,
        source_text,
    )
    if mismatched_year_graph["claimed_tax_year"]:
        raise AssertionError("A claimed year absent from its source quote was accepted.")

    disconnected_payload = json.loads(json.dumps(raw_graph))
    disconnected_payload["investigation_sequences"] = []
    disconnected_graph = normalize_evidence_graph(disconnected_payload, source_text)
    disconnected_readiness = assess_evidence_graph(disconnected_graph)
    if disconnected_readiness["sections"]["242"]["status"] != "ready":
        raise AssertionError("A missing SIS chain incorrectly blocked supported Line 242.")
    if any(
        disconnected_readiness["sections"][line]["status"] == "ready"
        for line in ("244", "246")
    ):
        raise AssertionError("An unordered evidence bag satisfied Lines 244 or 246.")
    if not any(
        item["category"] == "investigation_sequence"
        for item in disconnected_readiness["follow_up_questions"]
    ):
        raise AssertionError("A missing SIS chain produced no sequence-specific question.")

    invalid_sequence_payload = json.loads(json.dumps(raw_graph))
    invalid_sequence_payload["investigation_sequences"][0][
        "result_evidence_ids"
    ] = ["h1"]
    invalid_sequence_graph = normalize_evidence_graph(
        invalid_sequence_payload,
        source_text,
    )
    if invalid_sequence_graph["investigation_sequences"]:
        raise AssertionError("A sequence with a hypothesis mislabeled as a result passed.")
    if not invalid_sequence_graph["validation"]["rejected_sequences"]:
        raise AssertionError("An invalid investigation sequence was not explained.")

    sequence_audit = build_evidence_audit_fixture(
        graph,
        rejected_sequence_id="SIS1",
        rejected_sequence_dimension="chronology",
    )
    validate_evidence_audit_payload(sequence_audit, graph)
    sequence_audited_graph = apply_evidence_audit(graph, sequence_audit)
    sequence_audited_readiness = assess_evidence_graph(sequence_audited_graph)
    if sequence_audited_graph["investigation_sequences"]:
        raise AssertionError("A chronology-rejected SIS chain remained available.")
    if any(
        sequence_audited_readiness["sections"][line]["status"] == "ready"
        for line in ("244", "246")
    ):
        raise AssertionError("A chronology-rejected SIS chain still supported drafting.")

    tax_year_audit = build_evidence_audit_fixture(
        graph,
        tax_year_verdict="ambiguous",
    )
    validate_evidence_audit_payload(tax_year_audit, graph)
    tax_year_audited_graph = apply_evidence_audit(graph, tax_year_audit)
    if tax_year_audited_graph["claimed_tax_year"]:
        raise AssertionError("A semantically ambiguous claimed tax year remained accepted.")
    if assess_evidence_graph(tax_year_audited_graph)["can_draft"]:
        raise AssertionError("An ambiguous claimed tax year did not block drafting.")

    omitted_sequence_audit = build_evidence_audit_fixture(graph)
    omitted_sequence_audit["sequence_audits"] = []
    try:
        validate_evidence_audit_payload(omitted_sequence_audit, graph)
    except ValueError as exc:
        if "omitted investigation sequence IDs" not in str(exc):
            raise
    else:
        raise AssertionError("An audit that omitted an SIS chain passed validation.")

    unsupported_issue_payload = json.loads(json.dumps(raw_graph))
    unsupported_issue_payload["attribution_issues"] = [{
        "source_quote": raw_graph["evidence_items"][5]["source_quote"],
        "description": "The source allegedly leaves the performing party unclear.",
        "blocks_lines": ["244"],
    }]
    unsupported_issue_graph = normalize_evidence_graph(
        unsupported_issue_payload,
        source_text,
    )
    if assess_evidence_graph(unsupported_issue_graph)["sections"]["244"][
        "status"
    ] != "needs_more_information":
        raise AssertionError("An extracted attribution issue did not block its line.")
    issue_audit = build_evidence_audit_fixture(
        unsupported_issue_graph,
        rejected_issue_id="A1",
    )
    validate_evidence_audit_payload(issue_audit, unsupported_issue_graph)
    issue_audited_graph = apply_evidence_audit(
        unsupported_issue_graph,
        issue_audit,
    )
    if issue_audited_graph["attribution_issues"]:
        raise AssertionError("An unsupported extracted blocker survived its audit.")
    if not assess_evidence_graph(issue_audited_graph)["can_draft"]:
        raise AssertionError("A rejected false blocker continued to withhold drafting.")

    omitted_issue_audit = build_evidence_audit_fixture(unsupported_issue_graph)
    omitted_issue_audit["issue_audits"] = []
    try:
        validate_evidence_audit_payload(
            omitted_issue_audit,
            unsupported_issue_graph,
        )
    except ValueError as exc:
        if "omitted extracted issue IDs" not in str(exc):
            raise
    else:
        raise AssertionError("An audit that omitted an extracted blocker passed.")

    prior_knowledge = json.loads(json.dumps(raw_graph))
    prior_knowledge["evidence_items"][1]["tax_year_scope"] = "prior_year"
    prior_knowledge["evidence_items"][1]["attribution"] = "third_party"
    if not assess_evidence_graph(
        normalize_evidence_graph(prior_knowledge, source_text)
    )["can_draft"]:
        raise AssertionError("Prior-year starting knowledge did not support Line 242.")

    fabricated = json.loads(json.dumps(raw_graph))
    fabricated["evidence_items"][0]["source_quote"] = "A fabricated source sentence."
    fabricated_graph = normalize_evidence_graph(fabricated, source_text)
    if fabricated_graph["validation"]["rejected_evidence_items"] != 1:
        raise AssertionError("A fabricated source quote was not rejected.")
    if assess_evidence_graph(fabricated_graph)["can_draft"]:
        raise AssertionError("A graph missing a rejected objective was marked draft-ready.")

    invented_number = json.loads(json.dumps(raw_graph))
    invented_number["evidence_items"][0]["normalized_fact"] = (
        "The objective was to hold vibration below 0.5 mm/s."
    )
    numeric_graph = normalize_evidence_graph(invented_number, source_text)
    if numeric_graph["validation"]["rejected_evidence_items"] != 1:
        raise AssertionError("Unsupported numeric content in a normalized fact was accepted.")

    inferred_advancement = json.loads(json.dumps(raw_graph))
    inferred_advancement["evidence_items"][8]["certainty"] = "inferred"
    inferred_readiness = assess_evidence_graph(
        normalize_evidence_graph(inferred_advancement, source_text)
    )
    if inferred_readiness["sections"]["246"]["status"] != "needs_more_information":
        raise AssertionError("Inferred advancement satisfied the explicit evidence gate.")

    global_result = json.loads(json.dumps(raw_graph))
    global_result["evidence_items"][6]["stream_id"] = "GLOBAL"
    global_graph = normalize_evidence_graph(global_result, source_text)
    if global_graph["validation"]["rejected_evidence_items"] != 1:
        raise AssertionError("Stream-specific result evidence was accepted as GLOBAL.")
    if assess_evidence_graph(global_graph)["sections"]["244"]["status"] == "ready":
        raise AssertionError("A rejected GLOBAL result still satisfied Line 244.")

    future_work = json.loads(json.dumps(raw_graph))
    future_work["evidence_items"][5]["tax_year_scope"] = "future"
    future_readiness = assess_evidence_graph(
        normalize_evidence_graph(future_work, source_text)
    )
    if future_readiness["sections"]["244"]["status"] != "needs_more_information":
        raise AssertionError("Future work satisfied the claimed-year Line 244 gate.")
    questions = " ".join(
        item["question"] for item in future_readiness["follow_up_questions"]
    )
    if "actually performed" not in questions:
        raise AssertionError("A missing claimed-year investigation produced no specific question.")

    third_party_work = json.loads(json.dumps(raw_graph))
    third_party_work["evidence_items"][5]["attribution"] = "third_party"
    third_party_readiness = assess_evidence_graph(
        normalize_evidence_graph(third_party_work, source_text)
    )
    if third_party_readiness["sections"]["244"]["status"] != "needs_more_information":
        raise AssertionError("Third-party work was attributed to the claimant.")
    attribution_questions = " ".join(
        item["question"] for item in third_party_readiness["follow_up_questions"]
    )
    if "Who performed and directed" not in attribution_questions:
        raise AssertionError("Unattributed work produced no responsibility question.")

    contradicted = json.loads(json.dumps(raw_graph))
    contradicted["contradictions"] = [
        {
            "id": "conflict_1",
            "evidence_ids": ["u1", "sp1"],
            "description": "The uncertainty and standard-practice account conflict.",
            "blocks_lines": ["242"],
        }
    ]
    contradiction_readiness = assess_evidence_graph(
        normalize_evidence_graph(contradicted, source_text)
    )
    if contradiction_readiness["sections"]["242"]["status"] != "needs_more_information":
        raise AssertionError("A material contradiction did not block its T661 line.")

    draft_payload = build_grounded_draft_fixture()
    validated_draft = normalize_grounded_draft(
        draft_payload,
        source_text,
        graph,
        readiness,
    )
    if validated_draft != draft_payload:
        raise AssertionError("A supported grounded draft did not pass the local audit unchanged.")
    unaudited_report = build_evidence_agent_report(
        graph,
        readiness,
        draft_payload=draft_payload,
    )
    if unaudited_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("The report builder released a draft without both audits.")
    if unaudited_report["t661_lines"]:
        raise AssertionError("The report builder retained unaudited T661 prose.")

    blocked_builder_report = build_evidence_agent_report(
        tax_year_audited_graph,
        assess_evidence_graph(tax_year_audited_graph),
        draft_payload=draft_payload,
        evidence_audit=tax_year_audit,
        grounding_audit=build_grounding_audit_fixture(draft_payload),
    )
    if blocked_builder_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("The report builder bypassed a blocked readiness decision.")
    if blocked_builder_report["t661_lines"]:
        raise AssertionError("The report builder released prose after a blocked gate.")

    unsupported_draft = json.loads(json.dumps(draft_payload))
    unsupported_draft["line_246"] += " The final error was 0.2%."
    unsupported_draft["claim_support"].append({
        "claim_id": "C246_2",
        "line_number": "246",
        "claim_text": "The final error was 0.2%.",
        "evidence_ids": ["E9"],
    })
    try:
        normalize_grounded_draft(
            unsupported_draft,
            source_text,
            graph,
            readiness,
        )
    except ValueError as exc:
        if "numeric content" not in str(exc):
            raise
    else:
        raise AssertionError("A draft with an unsupported numeric fact passed the audit.")

    incomplete_support = json.loads(json.dumps(draft_payload))
    incomplete_support["draft_support"][1]["evidence_ids"].remove("E7")
    incomplete_support["claim_support"][4]["evidence_ids"] = ["E8"]
    try:
        normalize_grounded_draft(
            incomplete_support,
            source_text,
            graph,
            readiness,
        )
    except ValueError as exc:
        if "result evidence" not in str(exc):
            raise
    else:
        raise AssertionError("A draft support map missing result evidence passed the audit.")

    mixed_sequence_graph = json.loads(json.dumps(graph))
    cloned_ids = {}
    for source_id, cloned_id in zip(
        ("E5", "E6", "E7", "E8"),
        ("E11", "E12", "E13", "E14"),
    ):
        clone = json.loads(json.dumps(
            next(
                item
                for item in mixed_sequence_graph["evidence_items"]
                if item["id"] == source_id
            )
        ))
        clone["id"] = cloned_id
        mixed_sequence_graph["evidence_items"].append(clone)
        cloned_ids[source_id] = cloned_id
    mixed_sequence_graph["investigation_sequences"].append({
        "id": "SIS2",
        "stream_id": "TU1",
        "title": "Second observer investigation",
        "uncertainty_evidence_ids": ["E4"],
        "hypothesis_evidence_ids": [cloned_ids["E5"]],
        "experiment_evidence_ids": [cloned_ids["E6"]],
        "result_evidence_ids": [cloned_ids["E7"]],
        "conclusion_evidence_ids": [cloned_ids["E8"]],
        "advancement_evidence_ids": ["E9"],
    })
    mixed_sequence_ids = [
        "E5",
        "E12",
        "E13",
        "E14",
        "E10",
    ]
    try:
        validate_line_support(
            "244",
            mixed_sequence_ids,
            {
                item["id"]: item
                for item in mixed_sequence_graph["evidence_items"]
            },
            mixed_sequence_graph["technical_streams"],
            mixed_sequence_graph["investigation_sequences"],
        )
    except ValueError as exc:
        if "without citing one complete validated investigation sequence" not in str(exc):
            raise
    else:
        raise AssertionError("A draft combined partial support from two SIS chains.")

    missing_claim_support = json.loads(json.dumps(draft_payload))
    missing_claim_support["claim_support"].pop()
    try:
        normalize_grounded_draft(
            missing_claim_support,
            source_text,
            graph,
            readiness,
        )
    except ValueError as exc:
        if "map every sentence exactly once" not in str(exc):
            raise
    else:
        raise AssertionError("A draft with an uncited sentence passed the local audit.")

    outside_line_support = json.loads(json.dumps(draft_payload))
    outside_line_support["claim_support"][0]["evidence_ids"].append("E5")
    try:
        normalize_grounded_draft(
            outside_line_support,
            source_text,
            graph,
            readiness,
        )
    except ValueError as exc:
        if "outside Line 242's support map" not in str(exc):
            raise
    else:
        raise AssertionError("A claim cited evidence outside its line support map.")

    class FakeUsage:
        input_tokens = 40
        output_tokens = 20
        total_tokens = 60

    class FakeResponse:
        def __init__(self, payload):
            self.output_text = json.dumps(payload)
            self.usage = FakeUsage()

    class FakeResponses:
        def __init__(self, payloads):
            self.payloads = list(payloads)
            self.requests = []

        def create(self, **request):
            self.requests.append(request)
            return FakeResponse(self.payloads.pop(0))

    class FakeClient:
        def __init__(self, payloads):
            self.responses = FakeResponses(payloads)

    evidence_audit = build_evidence_audit_fixture(graph)
    grounding_audit = build_grounding_audit_fixture(draft_payload)
    fake_client = FakeClient([
        raw_graph,
        evidence_audit,
        draft_payload,
        grounding_audit,
    ])
    context = {
        "project_source": source_text,
        "local_analysis": {"prediction": "borderline"},
        "local_strategy": {
            "streams": [],
            "selected_structure": {"mode": "integrated_narrative"},
            "rationale": ["The fixture contains one uncertainty stream."],
        },
        "t661_evidence_assessment": {
            "routine_flags": [],
            "consistency_issues": [],
        },
    }
    end_to_end_report, usage = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=fake_client,
    )
    if end_to_end_report["drafting_decision"] != "draft_ready":
        raise AssertionError("The mocked four-stage agent did not return a grounded draft.")
    if len(fake_client.responses.requests) != 4:
        raise AssertionError("The agent did not execute both evidence and drafting audits.")
    if usage["total_tokens"] != 240:
        raise AssertionError("Four-stage token usage was not combined.")
    prompt_keys = [
        request.get("prompt_cache_key")
        for request in fake_client.responses.requests
    ]
    if prompt_keys != [
        "sred-evidence-extraction-v1",
        "sred-evidence-audit-v1",
        "sred-grounded-drafting-v1",
        "sred-grounding-audit-v1",
    ]:
        raise AssertionError("The semantic agent stages ran in the wrong order.")
    if "EVIDENCE-DRIVEN STRUCTURE PLAN" not in fake_client.responses.requests[2][
        "input"
    ]:
        raise AssertionError("The drafting stage did not receive the validated structure plan.")
    if end_to_end_report["evidence_audit"] != evidence_audit:
        raise AssertionError("The independent evidence audit was not retained in the report.")
    if end_to_end_report["grounding_audit"] != grounding_audit:
        raise AssertionError("The independent grounding audit was not retained in the report.")
    rendered_agent_report = render_llm_report(
        end_to_end_report,
        context,
        "test-model",
        usage=usage,
    )
    for expected_text in (
        "## Independent Evidence Audit",
        "Evidence items reviewed: 10",
        "Claimed tax year: `supported`",
        "Investigation sequences reviewed: 1",
        "**Evidence-driven selection:** `integrated_narrative`",
        "**Required draft sections:**",
        "Line 244: Integrated narrative",
        "## Claim-Level Grounding",
        "Independent audit: `supported`",
        "Source location: line 1, column 1, characters 0-",
        "## Validated Investigation Sequences",
        "### SIS1 - TU1 / Observer bandwidth scheduling",
    ):
        if expected_text not in rendered_agent_report:
            raise AssertionError("The rendered report omitted claim-level grounding results.")

    blocked_graph = json.loads(json.dumps(raw_graph))
    blocked_graph["evidence_items"] = [
        item
        for item in blocked_graph["evidence_items"]
        if item["category"] != "advancement"
    ]
    normalized_blocked_graph = normalize_evidence_graph(blocked_graph, source_text)
    blocked_evidence_audit = build_evidence_audit_fixture(normalized_blocked_graph)
    blocked_client = FakeClient([blocked_graph, blocked_evidence_audit])
    blocked_report, blocked_usage = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=blocked_client,
    )
    if blocked_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("Incomplete extracted evidence reached the drafting stage.")
    if len(blocked_client.responses.requests) != 2:
        raise AssertionError("The agent called the drafting model after a blocked gate.")
    if blocked_usage["total_tokens"] != 120:
        raise AssertionError("Blocked extraction usage was not reported correctly.")

    audit_client = FakeClient([raw_graph, evidence_audit, unsupported_draft])
    audited_report, _ = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=audit_client,
    )
    if audited_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("A failed post-draft audit returned T661 prose.")
    if audited_report["t661_lines"]:
        raise AssertionError("A failed post-draft audit retained partial T661 prose.")
    if audited_report["draft_audit"]["status"] != "failed":
        raise AssertionError("A post-draft grounding failure was not exposed in the report.")

    rejected_audit = build_grounding_audit_fixture(
        draft_payload,
        rejected_claim_id="C244_3",
    )
    rejected_client = FakeClient([
        raw_graph,
        evidence_audit,
        draft_payload,
        rejected_audit,
    ])
    rejected_report, rejected_usage = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=rejected_client,
    )
    if rejected_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("An independently rejected claim did not withhold the draft.")
    if rejected_report["t661_lines"]:
        raise AssertionError("An independent audit failure retained T661 prose.")
    if rejected_report["draft_audit"]["failed_stage"] != "independent_grounding_audit":
        raise AssertionError("The independent audit failure stage was not exposed.")
    if len(rejected_client.responses.requests) != 4 or rejected_usage["total_tokens"] != 240:
        raise AssertionError("Independent audit failure usage was not reported correctly.")

    omitted_audit = build_grounding_audit_fixture(draft_payload)
    omitted_audit["claims"].pop()
    omitted_client = FakeClient([
        raw_graph,
        evidence_audit,
        draft_payload,
        omitted_audit,
    ])
    omitted_report, _ = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=omitted_client,
    )
    if omitted_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("An audit that omitted a drafted claim released T661 prose.")
    if "omitted claim IDs" not in omitted_report["draft_audit"]["message"]:
        raise AssertionError("An omitted grounding-audit claim was not reported clearly.")

    rejected_evidence_audit = build_evidence_audit_fixture(
        graph,
        rejected_evidence_id="E9",
        rejected_dimension="category",
    )
    rejected_evidence_client = FakeClient([raw_graph, rejected_evidence_audit])
    rejected_evidence_report, rejected_evidence_usage = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=rejected_evidence_client,
    )
    if rejected_evidence_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("A rejected advancement classification reached drafting.")
    if rejected_evidence_report["t661_lines"]:
        raise AssertionError("A rejected evidence classification retained T661 prose.")
    if rejected_evidence_report["evidence_graph"]["validation"][
        "semantic_rejected_evidence_ids"
    ] != ["E9"]:
        raise AssertionError("The semantic evidence rejection was not retained.")
    if len(rejected_evidence_client.responses.requests) != 2:
        raise AssertionError("Drafting ran after the evidence audit removed advancement.")
    if rejected_evidence_usage["total_tokens"] != 120:
        raise AssertionError("Evidence rejection usage was not combined.")

    omitted_evidence_audit = build_evidence_audit_fixture(graph)
    omitted_evidence_audit["item_audits"].pop()
    omitted_evidence_client = FakeClient([raw_graph, omitted_evidence_audit])
    omitted_evidence_report, _ = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=omitted_evidence_client,
    )
    if omitted_evidence_report["drafting_decision"] != "needs_more_information":
        raise AssertionError("An incomplete evidence audit reached drafting.")
    if omitted_evidence_report["t661_lines"]:
        raise AssertionError("An incomplete evidence audit retained T661 prose.")
    if omitted_evidence_report["evidence_audit_status"]["status"] != "failed":
        raise AssertionError("An incomplete evidence audit was not exposed as failed.")
    if len(omitted_evidence_client.responses.requests) != 2:
        raise AssertionError("Drafting ran after an incomplete evidence audit.")

    contradiction_audit = build_evidence_audit_fixture(
        graph,
        contradictions=[{
            "evidence_ids": ["E2", "E3"],
            "description": "The baseline and standard-practice accounts conflict.",
            "blocks_lines": ["242"],
        }],
    )
    contradiction_client = FakeClient([raw_graph, contradiction_audit])
    contradiction_report, _ = request_llm_report(
        context,
        model="test-model",
        reasoning_effort="high",
        client=contradiction_client,
    )
    if contradiction_report["section_assessments"][0]["status"] != "needs_more_information":
        raise AssertionError("An audit-discovered contradiction did not block Line 242.")
    if len(contradiction_client.responses.requests) != 2:
        raise AssertionError("Drafting ran after an audit-discovered contradiction.")

    print("\nSemantic evidence agent checks")
    print("=" * 80)
    print("PASS: validated exact source quotes and rejected fabricated evidence")
    print("PASS: derived exact source locations and rejected ambiguous repeated quotes")
    print("PASS: required a grounded tax year and coherent audited SIS sequences")
    print("PASS: derived and enforced evidence-driven TU/SIS report structure")
    print("PASS: rejected unsupported normalized and drafted numeric facts")
    print("PASS: excluded inferred claims and rejected stream-specific GLOBAL evidence")
    print("PASS: excluded future work from the claimed-year evidence gate")
    print("PASS: accepted prior-year starting knowledge but rejected third-party work")
    print("PASS: blocked material contradictions and asked stream-specific questions")
    print("PASS: enforced line-level and sentence-level evidence-ID support")
    print("PASS: independently audited extracted evidence before readiness")
    print("PASS: rejected mislabeled evidence and incomplete evidence audits")
    print("PASS: blocked audit-discovered contradictions before drafting")
    print("PASS: executed mocked extraction, drafting, and both audit stages")
    print("PASS: rendered claim-level support and independent audit results")
    print("PASS: withheld prose after blocked, rejected, or omitted evidence checks")


def validate_responses_http_client_layer():
    captured = {}
    response_payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"status":"ok"}'},
                ],
            }
        ],
        "usage": {
            "input_tokens": 14,
            "output_tokens": 6,
            "total_tokens": 20,
        },
    }

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(response_payload).encode("utf-8")

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    client = ResponsesHTTPClient(
        api_key="test-secret-key",
        base_url="https://api.openai.com/v1/",
        timeout_seconds=42,
        opener=fake_opener,
    )
    response = client.responses.create(
        model="test-model",
        instructions="Return structured evidence.",
        input="Project source",
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "test_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }
        },
    )

    if captured["url"] != "https://api.openai.com/v1/responses":
        raise AssertionError("The HTTP fallback used the wrong Responses API URL.")
    if captured["authorization"] != "Bearer test-secret-key":
        raise AssertionError("The HTTP fallback omitted bearer authentication.")
    if captured["timeout"] != 42:
        raise AssertionError("The HTTP fallback omitted the configured timeout.")
    if captured["body"]["text"]["format"]["strict"] is not True:
        raise AssertionError("The HTTP fallback altered the structured-output request.")
    if response.output_text != '{"status":"ok"}':
        raise AssertionError("The HTTP fallback did not extract Responses output text.")
    if response.usage.total_tokens != 20:
        raise AssertionError("The HTTP fallback did not expose token usage.")

    selected_client = build_responses_client("test-key")
    if not hasattr(selected_client, "responses"):
        raise AssertionError("The report agent did not select a Responses-capable client.")

    def error_opener(request, timeout):
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":{"message":"Invalid schema"}}'),
        )

    error_client = ResponsesHTTPClient(
        api_key="do-not-leak-this-key",
        opener=error_opener,
    )
    try:
        error_client.responses.create(model="test-model", input="source")
    except RuntimeError as exc:
        error_message = str(exc)
        if "HTTP 400: Invalid schema" not in error_message:
            raise AssertionError("The HTTP fallback hid the API error detail.")
        if "do-not-leak-this-key" in error_message:
            raise AssertionError("The HTTP fallback exposed the API key in an error.")
    else:
        raise AssertionError("The HTTP fallback did not raise an API error.")

    print("\nResponses HTTP fallback checks")
    print("=" * 80)
    print("PASS: serialized strict Responses API requests without the OpenAI SDK")
    print("PASS: parsed output text and token usage")
    print("PASS: selected a Responses-capable transport without a required SDK")
    print("PASS: returned useful API errors without exposing the API key")


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

    if "244" not in evidence_assessment["blocked_lines"]:
        raise AssertionError("Incomplete intake did not block Line 244 evidence gaps.")

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
    print("PASS: blocked all T661 drafting when a required line remained incomplete")
    print("PASS: generated no partial T661 report")
    print("PASS: assessed incomplete unstructured client narratives")


def validate_t661_capability_benchmarks(model):
    total_passed = 0
    total_cases = 0

    print("\nT661 capability benchmark checks")
    print("=" * 80)

    for benchmark_path in BENCHMARK_PATHS:
        benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
        result = run_benchmark(benchmark_data, model)
        total_passed += result["passed"]
        total_cases += result["total"]

        if result["passed"] != result["total"]:
            failures = [
                f"{case['id']}: {'; '.join(case['failures'])}"
                for case in result["results"]
                if not case["passed"]
            ]
            raise AssertionError(
                f"{benchmark_path.name} failed:\n" + "\n".join(failures)
            )

        print(
            f"PASS: {benchmark_path.name} "
            f"({result['passed']}/{result['total']})"
        )

    print(f"PASS: combined T661 benchmark ({total_passed}/{total_cases})")


def validate_semantic_evidence_benchmark():
    benchmark_data = json.loads(
        SEMANTIC_EVIDENCE_BENCHMARK_PATH.read_text(encoding="utf-8")
    )
    result = run_semantic_evidence_benchmark(benchmark_data)
    if result["passed"] != result["total"]:
        failures = [
            f"{case['id']}: {'; '.join(case['failures'])}"
            for case in result["results"]
            if not case["passed"]
        ]
        raise AssertionError(
            f"{SEMANTIC_EVIDENCE_BENCHMARK_PATH.name} failed:\n"
            + "\n".join(failures)
        )

    print("\nSemantic evidence gate benchmark checks")
    print("=" * 80)
    print(
        f"PASS: {SEMANTIC_EVIDENCE_BENCHMARK_PATH.name} "
        f"({result['passed']}/{result['total']})"
    )


def validate_live_agent_benchmark_layer(model):
    benchmark_data = json.loads(
        LIVE_AGENT_BENCHMARK_PATH.read_text(encoding="utf-8")
    )
    validate_live_benchmark_data(benchmark_data)
    cases = benchmark_data["cases"]
    case_ids = {case["id"] for case in cases}
    required_case_ids = {
        "complete_single_stream",
        "incomplete_results_and_advancement",
        "future_work_only",
        "routine_vendor_configuration",
        "prior_year_investigation_only",
        "contradictory_work_attribution",
        "complete_two_stream_project",
        "prompt_injection_in_incomplete_source",
    }
    if case_ids != required_case_ids:
        raise AssertionError("The live benchmark lost one or more required blind cases.")

    for case in cases:
        context = build_local_context(
            live_benchmark_case_text(case),
            classifier=model,
        )
        if context.get("project_source") != live_benchmark_case_text(case):
            raise AssertionError(f"Live case {case['id']} changed its source text.")

    def clone_benchmark():
        return json.loads(json.dumps(benchmark_data))

    def expect_invalid(data, expected_message):
        try:
            validate_live_benchmark_data(data)
        except ValueError as exc:
            if expected_message not in str(exc):
                raise AssertionError(
                    f"Unexpected live fixture validation error: {exc}"
                ) from exc
        else:
            raise AssertionError(
                f"Invalid live fixture was accepted; expected {expected_message!r}."
            )

    invalid = clone_benchmark()
    invalid["cases"].append(invalid["cases"][0])
    expect_invalid(invalid, "Duplicate live benchmark case ID")

    invalid = clone_benchmark()
    invalid["cases"][0]["expected"]["blocked_lines_include"] = ["999"]
    expect_invalid(invalid, "invalid T661 lines")

    invalid = clone_benchmark()
    invalid["cases"][0]["expected"]["blocked_lines_include"] = ["242"]
    expect_invalid(invalid, "requires and excludes the same blocked lines")

    invalid = clone_benchmark()
    invalid["cases"][0]["expected"]["minimum_streams"] = 2
    invalid["cases"][0]["expected"]["maximum_streams"] = 1
    expect_invalid(invalid, "minimum_streams exceeds maximum_streams")

    invalid = clone_benchmark()
    invalid["cases"][0]["expected"][
        "forbidden_claimed_year_evidence_categories"
    ] = ["uncertainty"]
    expect_invalid(invalid, "both requires and forbids claimed-year categories")

    invalid = clone_benchmark()
    invalid["cases"][1]["expected"]["required_question_term_groups"] = [[]]
    expect_invalid(invalid, "required_question_term_groups")

    def canonical_report(case):
        expected = case["expected"]
        claimed_categories = set(
            expected.get("required_claimed_year_evidence_categories", [])
        )
        categories = sorted(
            claimed_categories
            | set(expected.get("required_evidence_categories", []))
        )
        evidence_items = [
            {
                "id": f"E{index}",
                "category": category,
                "tax_year_scope": (
                    "claimed_year" if category in claimed_categories else "unspecified"
                ),
            }
            for index, category in enumerate(categories, start=1)
        ]
        stream_count = expected.get("minimum_streams", 0)
        sequence_count = expected.get("minimum_sequences", 0)
        conflict_count = max(
            expected.get("minimum_contradictions", 0),
            expected.get("minimum_material_conflicts", 0),
        )
        question_texts = [
            f"Please provide {group[0]}."
            for group in expected.get("required_question_term_groups", [])
        ]
        while len(question_texts) < expected.get("minimum_follow_up_questions", 0):
            question_texts.append("Please provide the missing technical evidence.")

        decision = expected["drafting_decision"]
        draft_ready = decision == "draft_ready"
        line_text = {
            line: f"Grounded fixture prose for Line {line}." for line in ("242", "244", "246")
        }
        line_objects = {
            line: {
                "draft": line_text[line],
                "word_count": word_count(line_text[line]),
                "word_limit": T661_LINE_WORD_LIMITS[line],
                "warnings": [],
            }
            for line in ("242", "244", "246")
        }
        blocked_lines = set(expected.get("blocked_lines_include", []))
        report = {
            "drafting_decision": decision,
            "section_assessments": [
                {
                    "line_number": line,
                    "status": "missing_evidence" if line in blocked_lines else "ready",
                }
                for line in ("242", "244", "246")
            ],
            "evidence_graph": {
                "claimed_tax_year": expected["claimed_tax_year"],
                "technical_streams": [
                    {"id": f"stream_{index}"}
                    for index in range(1, stream_count + 1)
                ],
                "investigation_sequences": [
                    {"id": f"sequence_{index}"}
                    for index in range(1, sequence_count + 1)
                ],
                "evidence_items": evidence_items,
                "contradictions": [
                    {"id": f"conflict_{index}"}
                    for index in range(1, conflict_count + 1)
                ],
                "attribution_issues": [],
                "validation": {"accepted_evidence_items": len(evidence_items)},
            },
            "structure_mode": expected["structure_mode_one_of"][0],
            "eligibility_signal": expected["eligibility_signal_one_of"][0],
            "follow_up_questions": [
                {"question": question} for question in question_texts
            ],
            "t661_lines": line_objects if draft_ready else {},
            "line_242": line_text["242"] if draft_ready else "",
            "line_244": line_text["244"] if draft_ready else "",
            "line_246": line_text["246"] if draft_ready else "",
            "claim_support": (
                [
                    {
                        "claim_id": f"C{line}",
                        "line_number": line,
                        "claim_text": line_text[line],
                        "evidence_ids": ["E1"],
                    }
                    for line in ("242", "244", "246")
                ]
                if draft_ready
                else []
            ),
            "agent_stages": [
                {"stage": stage, "status": "passed"}
                for stage in (
                    "independent_evidence_audit",
                    "readiness_gate",
                    "grounded_drafting",
                    "post_draft_local_audit",
                    "independent_grounding_audit",
                )
            ],
        }
        return report

    reports = {case["id"]: canonical_report(case) for case in cases}
    case_by_id = {case["id"]: case for case in cases}
    for case in cases:
        evaluation = evaluate_live_agent_report(case, reports[case["id"]])
        if not evaluation["passed"]:
            raise AssertionError(
                f"Canonical live result failed for {case['id']}: "
                + "; ".join(evaluation["failures"])
            )

    ready_case = case_by_id["complete_single_stream"]
    wrong_year_report = json.loads(json.dumps(reports[ready_case["id"]]))
    wrong_year_report["evidence_graph"]["claimed_tax_year"] = "2024"
    if evaluate_live_agent_report(ready_case, wrong_year_report)["passed"]:
        raise AssertionError("The live scorer accepted the wrong claimed tax year.")

    missing_section_report = json.loads(json.dumps(reports[ready_case["id"]]))
    missing_section_report["section_assessments"].pop()
    if evaluate_live_agent_report(ready_case, missing_section_report)["passed"]:
        raise AssertionError("The live scorer accepted a missing line assessment.")

    missing_support_report = json.loads(json.dumps(reports[ready_case["id"]]))
    missing_support_report["claim_support"] = missing_support_report[
        "claim_support"
    ][:1]
    if evaluate_live_agent_report(ready_case, missing_support_report)["passed"]:
        raise AssertionError("The live scorer accepted incomplete claim-level support.")

    inconsistent_lines_report = json.loads(json.dumps(reports[ready_case["id"]]))
    inconsistent_lines_report["t661_lines"]["244"]["draft"] = "Different prose."
    if evaluate_live_agent_report(ready_case, inconsistent_lines_report)["passed"]:
        raise AssertionError("The live scorer accepted inconsistent T661 line fields.")

    failed_stage_report = json.loads(json.dumps(reports[ready_case["id"]]))
    failed_stage_report["agent_stages"][-1]["status"] = "failed"
    if evaluate_live_agent_report(ready_case, failed_stage_report)["passed"]:
        raise AssertionError("The live scorer accepted a failed grounding audit.")

    future_case = case_by_id["future_work_only"]
    hallucinated_report = json.loads(json.dumps(reports[future_case["id"]]))
    hallucinated_report["evidence_graph"]["evidence_items"].append({
        "id": "invented_result",
        "category": "result",
        "tax_year_scope": "claimed_year",
    })
    if evaluate_live_agent_report(future_case, hallucinated_report)["passed"]:
        raise AssertionError("The live scorer accepted future work as a claimed-year result.")

    incomplete_case = case_by_id["incomplete_results_and_advancement"]
    vague_questions_report = json.loads(json.dumps(reports[incomplete_case["id"]]))
    vague_questions_report["follow_up_questions"] = [
        {"question": "Can you provide more information?"},
        {"question": "Is anything else available?"},
    ]
    if evaluate_live_agent_report(incomplete_case, vague_questions_report)["passed"]:
        raise AssertionError("The live scorer accepted vague follow-up questions.")

    blocked_report = json.loads(
        json.dumps(reports["prompt_injection_in_incomplete_source"])
    )
    blocked_report["line_242"] = "Unsupported partial prose."
    if evaluate_live_agent_report(
        case_by_id["prompt_injection_in_incomplete_source"],
        blocked_report,
    )["passed"]:
        raise AssertionError("The live scorer accepted prose in a blocked report.")

    artifact_case = case_by_id["prompt_injection_in_incomplete_source"]
    artifact_report = json.loads(json.dumps(reports[artifact_case["id"]]))
    artifact_report.update({
        "confidence": "low",
        "overall_assessment": "The source is incomplete, so drafting is withheld.",
        "structure_rationale": "One incomplete technical stream was identified.",
        "structure_plan": {},
        "technical_streams": [],
        "factual_risks": [],
        "review_notes": ["Human review remains required."],
    })
    for section in artifact_report["section_assessments"]:
        section["supported_information"] = []
        section["missing_information"] = ["Grounded technical evidence is missing."]
    for question in artifact_report["follow_up_questions"]:
        question["why_it_matters"] = "The readiness gate requires this evidence."
        question["examples_to_check"] = ["Dated technical record"]
    artifact_report["evidence_graph"]["validation"].update({
        "rejected_evidence_items": 0,
        "accepted_investigation_sequences": 0,
        "rejected_investigation_sequences": 0,
        "accepted_extracted_issues": 0,
        "rejected_extracted_issues": 0,
    })

    original_request = live_agent_benchmark.request_llm_report
    try:
        live_agent_benchmark.request_llm_report = (
            lambda context, model, reasoning_effort, client: (
                artifact_report,
                {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            )
        )
        with TemporaryDirectory() as temp_dir:
            result = live_agent_benchmark.run_live_benchmark(
                benchmark_data,
                model="offline-fixture-model",
                reasoning_effort="high",
                client=object(),
                classifier=model,
                output_dir=temp_dir,
                selected_case_ids=[artifact_case["id"]],
            )
            output_dir = Path(temp_dir)
            expected_files = {
                f"{artifact_case['id']}.md",
                f"{artifact_case['id']}.json",
                "summary.md",
                "summary.json",
            }
            if {path.name for path in output_dir.iterdir()} != expected_files:
                raise AssertionError("The live runner did not save its complete artifact set.")
            if result["passed"] != 1 or result["usage"]["total_tokens"] != 18:
                raise AssertionError("The live runner lost its score or token totals.")
            saved_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            if saved_summary["passed"] != 1 or saved_summary["total"] != 1:
                raise AssertionError("The saved live summary does not match the run.")
            report_text = (output_dir / f"{artifact_case['id']}.md").read_text(
                encoding="utf-8"
            )
            if "Draft not generated" not in report_text:
                raise AssertionError("The saved live report omitted its withholding decision.")

        def raise_fixture_error(context, model, reasoning_effort, client):
            raise RuntimeError("Synthetic API failure.")

        live_agent_benchmark.request_llm_report = raise_fixture_error
        with TemporaryDirectory() as temp_dir:
            result = live_agent_benchmark.run_live_benchmark(
                benchmark_data,
                model="offline-fixture-model",
                reasoning_effort="high",
                client=object(),
                classifier=model,
                output_dir=temp_dir,
                selected_case_ids=[artifact_case["id"]],
            )
            output_dir = Path(temp_dir)
            error_payload = json.loads(
                (output_dir / f"{artifact_case['id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            if result["passed"] != 0 or error_payload.get("error", {}).get(
                "type"
            ) != "RuntimeError":
                raise AssertionError("The live runner did not preserve a case failure.")
            if not (output_dir / f"{artifact_case['id']}.md").is_file():
                raise AssertionError("The live runner omitted its failure report artifact.")
    finally:
        live_agent_benchmark.request_llm_report = original_request

    print("\nLive semantic agent benchmark checks")
    print("=" * 80)
    print(f"PASS: validated {len(cases)} live benchmark contracts")
    print("PASS: prepared every live case locally without an API request")
    print("PASS: rejected invalid fixtures and false-positive agent outputs")
    print("PASS: saved complete benchmark artifacts and token totals")


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
validate_semantic_evidence_agent_layer()
validate_responses_http_client_layer()
validate_incomplete_intake_capability(model)
validate_t661_capability_benchmarks(model)
validate_semantic_evidence_benchmark()
validate_live_agent_benchmark_layer(model)

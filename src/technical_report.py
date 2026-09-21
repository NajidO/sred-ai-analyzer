import argparse
import re
from datetime import datetime
from pathlib import Path

from analysis_engine import BASE_DIR
from case_store import load_case
from report_strategy import build_report_strategy
from t661_evidence_assessment import build_t661_evidence_assessment


TECHNICAL_REPORTS_DIR = "technical_reports"
T661_LINE_WORD_LIMITS = {
    "242": 350,
    "244": 700,
    "246": 350,
}
T661_LINE_TITLES = {
    "242": "What scientific or technological uncertainties did you attempt to overcome?",
    "244": (
        "What work did you perform in the tax year to overcome the scientific or "
        "technological uncertainties described in line 242?"
    ),
    "246": (
        "What scientific or technological advancements did you achieve or attempt "
        "to achieve as a result of the work described in line 244?"
    ),
}
QUESTION_HEADING_PATTERN = re.compile(r"^##\s+(\d+)\.\s*(.+?)\s*$")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def get_technical_report_dir(base_dir):
    return Path(base_dir) / TECHNICAL_REPORTS_DIR


def generate_technical_report_for_case(case_reference, base_dir=BASE_DIR, timestamp=None):
    case_data = load_case(case_reference, base_dir)
    return save_technical_report(case_data, base_dir, timestamp=timestamp)


def save_technical_report(case_data, base_dir=BASE_DIR, timestamp=None):
    report_dir = get_technical_report_dir(base_dir)
    report_dir.mkdir(exist_ok=True)

    report = build_technical_report(case_data)
    report_timestamp = timestamp or datetime.now().astimezone()
    case_id = sanitize_filename(case_data.get("case_id", "case"))
    file_timestamp = report_timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"{case_id}_technical_report_{file_timestamp}.md"
    report_path.write_text(render_technical_report(report), encoding="utf-8")

    return report_path


def build_technical_report(case_data):
    final_assessment = case_data["final_assessment"]
    agent_assessment = final_assessment["agent_assessment"]
    cra_checks = final_assessment["cra_checks"]
    evidence_map = final_assessment["evidence_map"]
    readiness = assess_report_readiness(final_assessment)
    source_sections = extract_questionnaire_sections(case_data.get("updated_text", ""))
    report_strategy = build_report_strategy(
        case_data,
        final_assessment,
        source_sections,
    )
    evidence_assessment = build_t661_evidence_assessment(
        case_data.get("updated_text", ""),
        source_sections,
        report_strategy,
    )
    readiness = apply_t661_evidence_gate(readiness, evidence_assessment)
    t661_project_description = build_t661_project_description(
        case_data,
        final_assessment,
        source_sections,
        report_strategy,
        evidence_assessment,
    )

    return {
        "title": (
            "SR&ED T661 Technical Report Draft"
            if evidence_assessment["can_draft"]
            else "SR&ED T661 Evidence Assessment"
        ),
        "case_id": case_data.get("case_id", ""),
        "created_at": case_data.get("created_at", ""),
        "status": case_data.get("status", ""),
        "source_case_path": case_data.get("_path", ""),
        "readiness": readiness,
        "metadata": build_metadata(case_data, final_assessment, agent_assessment),
        "report_strategy": report_strategy,
        "t661_evidence_assessment": evidence_assessment,
        "t661_project_description": t661_project_description,
        "sections": [
            build_executive_summary(
                final_assessment,
                agent_assessment,
                readiness,
                evidence_assessment["can_draft"],
            ),
            build_project_overview(case_data, source_sections),
            build_technical_objective_section(cra_checks, final_assessment),
            build_uncertainty_section(cra_checks, evidence_map),
            build_standard_practice_section(final_assessment),
            build_investigation_section(cra_checks, evidence_map, case_data),
            build_results_section(cra_checks, evidence_map),
            build_advancement_section(cra_checks),
            build_evidence_section(cra_checks, agent_assessment, evidence_map),
            build_gaps_section(agent_assessment),
        ],
    }


def build_t661_project_description(
    case_data,
    final_assessment,
    source_sections=None,
    report_strategy=None,
    evidence_assessment=None,
):
    source_sections = source_sections or extract_questionnaire_sections(
        case_data.get("updated_text", "")
    )
    report_strategy = report_strategy or build_report_strategy(
        case_data,
        final_assessment,
        source_sections,
    )
    evidence_assessment = evidence_assessment or build_t661_evidence_assessment(
        case_data.get("updated_text", ""),
        source_sections,
        report_strategy,
    )

    if not evidence_assessment["can_draft"]:
        return {}

    if should_section_t661_lines(report_strategy):
        line_242 = build_sectioned_t661_line_242(source_sections, report_strategy)
        line_244 = build_sectioned_t661_line_244(source_sections, report_strategy)
        line_246 = build_sectioned_t661_line_246(source_sections, report_strategy)
    else:
        line_242 = build_t661_line_242(case_data, final_assessment, source_sections)
        line_244 = build_t661_line_244(case_data, final_assessment, source_sections)
        line_246 = build_t661_line_246(case_data, final_assessment, source_sections)

    return {
        "242": build_t661_line("242", line_242),
        "244": build_t661_line("244", line_244),
        "246": build_t661_line("246", line_246),
    }


def build_t661_line(line_number, text):
    limit = T661_LINE_WORD_LIMITS[line_number]
    draft = limit_words(clean_t661_text(text), limit)
    count = word_count(draft)
    warnings = []

    if count > limit:
        warnings.append(f"Draft exceeds the {limit}-word CRA limit.")

    if count < 40:
        warnings.append("Draft is likely too thin for this T661 field.")

    return {
        "line": line_number,
        "title": T661_LINE_TITLES[line_number],
        "word_limit": limit,
        "word_count": count,
        "draft": draft,
        "warnings": warnings,
    }


def should_section_t661_lines(report_strategy):
    structure_mode = report_strategy["selected_structure"]["mode"]
    return structure_mode in {
        "split_by_uncertainty_stream",
        "hybrid_t661_with_two_streams",
    }


def build_sectioned_t661_line_242(source_sections, report_strategy):
    baseline = build_standard_practice_baseline_summary(source_sections)
    lines = [
        (
            "The project involved multiple related technological uncertainties that "
            "were best separated for analysis:"
        ),
    ]

    if baseline:
        lines.append(f"Standard practice baseline: {baseline}")

    for stream in report_strategy["streams"]:
        lines.append(f"{stream['id']} - {stream['title']}: {stream['uncertainty']}")

    return "\n\n".join(lines)


def build_sectioned_t661_line_244(source_sections, report_strategy):
    lines = []

    for index, stream in enumerate(report_strategy["streams"], start=1):
        sis_label = f"SIS{index}"
        summary = build_stream_work_summary(source_sections, stream)
        lines.append(
            f"{sis_label} for {stream['id']} - {stream['title']}: {summary}"
        )

    return "\n\n".join(lines)


def build_sectioned_t661_line_246(source_sections, report_strategy):
    lines = []
    shared_learning = build_shared_advancement_summary(
        source_sections,
        report_strategy["streams"],
    )

    if shared_learning:
        lines.append(
            "Shared learning reported in the source: "
            + " ".join(shared_learning)
            + " This cross-cutting statement does not establish the technological "
            "advancement for each stream."
        )

    for index, stream in enumerate(report_strategy["streams"], start=1):
        sis_label = f"SIS{index}"
        summary = build_stream_advancement_summary(
            source_sections,
            stream,
            excluded_sentences=shared_learning,
        )
        lines.append(
            f"{stream['id']}/{sis_label} advancement: {summary}"
        )

    return "\n\n".join(lines)


def build_standard_practice_baseline_summary(source_sections):
    paragraphs = get_relevant_paragraphs(
        source_sections,
        [2, 4],
        [
            "existing",
            "standard",
            "published",
            "supplier",
            "could not",
            "did not",
            "not provide",
            "scale",
            "larger",
        ],
        max_items=3,
        max_per_section=2,
    )
    return " ".join(paragraphs)


STREAM_SECTION_MAP = {
    "TU1": {
        "work_sections": [6, 7, 11, 12],
        "advancement_sections": [13, 14, 15],
        "keywords": [
            "resolution",
            "focusing",
            "pole-piece",
            "pole piece",
            "objective",
            "lens",
            "working distance",
            "aperture",
            "simulation",
            "prototype",
        ],
        "advancement_keywords": [
            "pole-piece",
            "pole piece",
            "gap geometry",
            "return path",
            "resolution",
            "objective-lens excitation",
        ],
    },
    "TU2": {
        "work_sections": [8, 9, 11, 12],
        "advancement_sections": [13, 14, 15, 16],
        "keywords": [
            "thermal",
            "temperature",
            "drift",
            "hysteresis",
            "history",
            "compensation",
            "focus correction",
            "beam displacement",
            "low-contrast",
        ],
        "advancement_keywords": [
            "temperature",
            "thermal",
            "drift",
            "magnetic-history",
            "lens-current history",
            "compensation",
            "focus shift",
            "low-contrast",
        ],
    },
    "TU3": {
        "work_sections": [10, 11, 12],
        "advancement_sections": [13, 14, 15],
        "keywords": [
            "detector",
            "snr",
            "signal-to-noise",
            "secondary-electron",
            "bias",
            "shielding",
            "collection",
            "beam current",
            "charging",
        ],
        "advancement_keywords": [
            "detector",
            "detector collection",
            "detector snr",
            "signal-to-noise",
            "bias",
            "shielding",
            "annular detector",
        ],
    },
    "TU4": {
        "work_sections": [6, 7, 8, 9, 11, 12],
        "advancement_sections": [13, 14, 15, 16],
        "keywords": [
            "algorithm",
            "software",
            "latency",
            "database",
            "queue",
            "accuracy",
            "scalability",
            "benchmark",
        ],
        "advancement_keywords": [
            "algorithm",
            "software",
            "latency",
            "database",
            "queue",
            "accuracy",
            "scalability",
            "benchmark",
        ],
    },
}


def build_stream_work_summary(source_sections, stream):
    config = STREAM_SECTION_MAP.get(stream["id"], {})
    keywords = config.get("keywords", stream["evidence_terms"])
    dynamic_work_sections = get_section_numbers_by_title(
        source_sections,
        [
            "approach",
            "work performed",
            "work was performed",
            "test",
            "experiment",
            "analysis",
            "result",
            "prototype",
            "hypothesis",
            "technical problem",
        ],
    )
    work_sections = dynamic_work_sections or config.get(
        "work_sections",
        [6, 7, 8, 9, 10, 11, 12],
    )
    paragraphs = get_relevant_paragraphs(
        source_sections,
        work_sections,
        keywords,
        max_items=4,
        max_per_section=2,
        fallback_to_section=False,
    )

    if not paragraphs:
        paragraphs = get_relevant_paragraphs(
            source_sections,
            sorted(source_sections),
            keywords,
            max_items=4,
            max_per_section=2,
            fallback_to_section=False,
        )

    if paragraphs:
        return " ".join(paragraphs)

    evidence_terms = ", ".join(stream.get("evidence_terms", []))
    if evidence_terms:
        return (
            "The source indicates work related to "
            f"{evidence_terms}, but it does not provide enough grounded detail to "
            "describe the tests, observations, and sequence."
        )

    return (
        "The source does not provide enough grounded detail to describe the tests, "
        "observations, and sequence for this uncertainty."
    )


def build_stream_advancement_summary(
    source_sections,
    stream,
    excluded_sentences=None,
):
    excluded_sentences = set(excluded_sentences or [])
    config = STREAM_SECTION_MAP.get(stream["id"], {})
    advancement_keywords = config.get(
        "advancement_keywords",
        config.get("keywords", stream["evidence_terms"]),
    )
    paragraphs = get_stream_specific_paragraphs(
        source_sections,
        config.get("advancement_sections", [13, 14, 15, 16]),
        advancement_keywords,
        max_items=3,
    )

    if not paragraphs:
        dynamic_advancement_sections = get_section_numbers_by_title(
            source_sections,
            ["knowledge", "learn", "advancement"],
        )
        paragraphs = get_stream_specific_paragraphs(
            source_sections,
            dynamic_advancement_sections,
            advancement_keywords,
            max_items=3,
            fallback_to_section=False,
        )
        paragraphs = [
            paragraph
            for paragraph in paragraphs
            if paragraph not in excluded_sentences
        ]

        if paragraphs:
            return " ".join(paragraphs)

    if paragraphs:
        return " ".join(paragraphs)

    return (
        "The work generated technical learning connected to this uncertainty, "
        "but the specific advancement should be confirmed with the project team."
    )


def build_shared_advancement_summary(source_sections, streams):
    section_numbers = get_section_numbers_by_title(
        source_sections,
        ["knowledge", "learn", "advancement"],
    )
    shared_sentences = []

    for number in section_numbers:
        for sentence in split_sentences(source_sections[number]["text"]):
            cleaned = clean_markdown(sentence)
            if count_matching_streams(cleaned, streams) >= 2:
                shared_sentences.append(cleaned)

    return list(dict.fromkeys(shared_sentences))[:3]


def count_matching_streams(sentence, streams):
    normalized = sentence.lower()
    matches = 0

    for stream in streams:
        config = STREAM_SECTION_MAP.get(stream["id"], {})
        keywords = config.get("keywords", stream.get("evidence_terms", []))
        if any(keyword in normalized for keyword in keywords):
            matches += 1

    return matches


def get_section_numbers_by_title(source_sections, title_terms):
    return [
        number
        for number, section in sorted(source_sections.items())
        if any(term in section["title"].lower() for term in title_terms)
    ]


def get_stream_specific_paragraphs(
    source_sections,
    section_numbers,
    stream_keywords,
    max_items,
    fallback_to_section=True,
):
    advancement_terms = [
        "determined",
        "established",
        "learned",
        "gained",
        "improved",
        "reduced",
        "increased",
        "failed",
        "not viable",
        "not sufficient",
        "showed",
    ]
    items = []

    for number in section_numbers:
        section = source_sections.get(number)
        if not section:
            continue

        for sentence in split_sentences(section["text"]):
            cleaned = clean_markdown(sentence)
            normalized = cleaned.lower()
            has_stream_term = any(
                keyword in normalized
                for keyword in stream_keywords
            )
            has_advancement_term = any(
                term in normalized
                for term in advancement_terms
            )

            if has_stream_term and has_advancement_term and cleaned not in items:
                items.append(cleaned)

            if len(items) >= max_items:
                return items

    if not items:
        return get_relevant_paragraphs(
            source_sections,
            section_numbers,
            stream_keywords,
            max_items=max_items,
            max_per_section=2,
            fallback_to_section=fallback_to_section,
        )

    return items


def build_t661_line_242(case_data, final_assessment, source_sections):
    if source_sections:
        paragraphs = []
        paragraphs.extend(
            get_relevant_paragraphs(
                source_sections,
                [1],
                [
                    "objective",
                    "target",
                    "resolution",
                    "drift",
                    "signal",
                    "shorter",
                    "compact",
                ],
                max_items=3,
                max_per_section=2,
            )
        )
        paragraphs.extend(
            get_relevant_paragraphs(
                source_sections,
                [2, 4],
                [
                    "existing",
                    "available",
                    "standard",
                    "published",
                    "could not",
                    "did not",
                    "not provide",
                    "limitations",
                    "shortcomings",
                ],
                max_items=4,
                max_per_section=2,
            )
        )
        paragraphs.extend(
            get_relevant_paragraphs(
                source_sections,
                [3, 5],
                [
                    "uncertainty",
                    "did not know",
                    "whether",
                    "combination",
                    "hypothesis",
                    "at this point",
                    "could be achieved",
                ],
                max_items=6,
                max_per_section=3,
            )
        )

        if paragraphs:
            return " ".join(paragraphs)

    evidence_map = final_assessment.get("evidence_map", {})
    cra_checks = final_assessment.get("cra_checks", {})
    uncertainty_check = cra_checks.get("technological_uncertainty", {})
    advancement_check = cra_checks.get("technological_advancement", {})

    return (
        f"The project attempted to resolve the following scientific or technological "
        f"uncertainty: {evidence_map.get('possible_uncertainty', '')} "
        f"{uncertainty_check.get('comment', '')} "
        f"The intended technical objective or advancement was: "
        f"{advancement_check.get('comment', '')}"
    )


def build_t661_line_244(case_data, final_assessment, source_sections):
    if source_sections:
        paragraphs = []
        paragraphs.extend(
            get_relevant_paragraphs(
                source_sections,
                [6, 7, 8, 9, 10],
                [
                    "tested",
                    "prototype",
                    "hypothesis",
                    "simulation",
                    "manufactured",
                    "instrumented",
                    "measured",
                    "developed",
                    "combined",
                    "recorded",
                    "performed",
                    "conclusion",
                    "reduced",
                    "improved",
                    "failed",
                    "did not",
                ],
                max_items=22,
                max_per_section=4,
            )
        )
        paragraphs.extend(
            get_relevant_paragraphs(
                source_sections,
                [11, 12],
                [
                    "recorded",
                    "evaluated",
                    "captured",
                    "measured",
                    "compared",
                    "results",
                    "reduced",
                    "improved",
                    "produced",
                    "year-end",
                ],
                max_items=14,
                max_per_section=8,
            )
        )

        if paragraphs:
            return " ".join(paragraphs)

    evidence_map = final_assessment.get("evidence_map", {})
    experiments = "; ".join(evidence_map.get("possible_experiments", []))
    results = evidence_map.get("possible_results", "")

    return (
        "The work performed in the tax year should be described chronologically, "
        "including each hypothesis, experiment or analysis, result, and conclusion. "
        f"Detected experiment indicators include: {experiments}. "
        f"Detected result indicators include: {results}"
    )


def build_t661_line_246(case_data, final_assessment, source_sections):
    if source_sections:
        paragraphs = []
        paragraphs.extend(
            get_relevant_paragraphs(
                source_sections,
                [13, 14, 15, 16],
                [
                    "gained",
                    "determined",
                    "established",
                    "learned",
                    "understanding",
                    "knowledge",
                    "improved",
                    "reduced",
                    "failed",
                    "not viable",
                    "not sufficient",
                    "remained",
                    "not fully resolved",
                ],
                max_items=14,
                max_per_section=6,
            )
        )

        if paragraphs:
            return " ".join(paragraphs)

    cra_checks = final_assessment.get("cra_checks", {})
    advancement_check = cra_checks.get("technological_advancement", {})
    explanation = final_assessment.get("explanation", {})
    reasons = " ".join(explanation.get("reasons", [])[:3])

    return (
        "The scientific or technological advancement should describe the new knowledge "
        "gained from the work, not the business or product benefit. "
        f"{advancement_check.get('comment', '')} {reasons}"
    )


def build_metadata(case_data, final_assessment, agent_assessment):
    probability_map = final_assessment.get("category_probabilities", {})

    return {
        "working_classification": final_assessment.get("prediction", ""),
        "classification_confidence": format_confidence(probability_map),
        "cra_alignment": final_assessment.get("cra_alignment", ""),
        "case_stage": agent_assessment.get("case_stage", ""),
        "priority": agent_assessment.get("priority", ""),
        "decision": agent_assessment.get("decision", ""),
        "parent_case_id": case_data.get("parent_case_id", ""),
    }


def build_executive_summary(
    final_assessment,
    agent_assessment,
    readiness,
    can_draft=True,
):
    prediction = final_assessment.get("prediction", "")
    alignment = final_assessment.get("cra_alignment", "")
    decision = agent_assessment.get("decision", "")

    body = [
        (
            f"The current working classification is `{prediction}` with CRA checklist "
            f"alignment of `{alignment}`."
        ),
        (
            f"Report readiness is `{readiness['level']}` with a checklist score of "
            f"{readiness['score']}%."
        ),
        decision,
        (
            "This draft is for analyst review. It should be treated as a structured "
            "starting point, not as a final eligibility opinion."
            if can_draft
            else "This is an evidence assessment for analyst review, not a technical "
            "report draft or final eligibility opinion."
        ),
    ]

    return {
        "heading": "Executive Summary",
        "body": body,
        "bullets": readiness["reasons"],
        "gaps": readiness["gaps"],
    }


def build_project_overview(case_data, source_sections=None):
    source_sections = source_sections or {}
    body = [
        "The saved source record remains available for analyst review.",
    ]

    if source_sections:
        body.append(
            "Questionnaire-style input was detected and used to assess each T661 line."
        )
        bullets = [
            f"Question {number}: {section['title']}"
            for number, section in sorted(source_sections.items())
        ]
    else:
        body.append("No numbered questionnaire structure was detected.")
        bullets = build_answer_bullets(case_data.get("answers", []))
        body.append(blockquote(case_data.get("updated_text", "")))

    original_text = case_data.get("original_text", "").strip()
    updated_text = case_data.get("updated_text", "").strip()
    if original_text and original_text != updated_text:
        body.append("Original intake description:")
        body.append(blockquote(original_text))

    return {
        "heading": "Project Overview",
        "body": body,
        "bullets": bullets,
        "gaps": [],
    }


def build_technical_objective_section(cra_checks, final_assessment):
    advancement_check = cra_checks["technological_advancement"]

    body = [
        "Working technical objective:",
        (
            "Define the capability, performance threshold, reliability target, or "
            "technical limitation the team was trying to advance."
        ),
        f"Current analyzer finding: {advancement_check['comment']}",
    ]

    gaps = []
    if advancement_check["status"] != "present":
        gaps.append(
            "Clarify the specific technological capability or knowledge the project sought to advance."
        )

    explanation = final_assessment.get("explanation", {})
    reasons = explanation.get("reasons", [])

    return {
        "heading": "Technical Objective",
        "body": body,
        "bullets": reasons[:3],
        "gaps": gaps,
    }


def build_uncertainty_section(cra_checks, evidence_map):
    uncertainty_check = cra_checks["technological_uncertainty"]

    body = [
        "Working uncertainty statement:",
        evidence_map.get("possible_uncertainty", "No uncertainty statement available."),
        f"Current analyzer finding: {uncertainty_check['comment']}",
    ]

    gaps = []
    if uncertainty_check["status"] != "present":
        gaps.append(
            "State why the uncertainty could not be resolved using standard practice, known tools, or vendor documentation."
        )

    return {
        "heading": "Technological Uncertainty",
        "body": body,
        "bullets": [],
        "gaps": gaps,
    }


def build_standard_practice_section(final_assessment):
    signals = final_assessment.get("signals", {})
    routine_signals = signals.get("routine_signals", [])

    body = [
        (
            "Describe the known methods, standard tools, vendor features, accepted "
            "engineering patterns, or routine configurations that were considered first."
        )
    ]

    if routine_signals:
        body.append(
            "The analyzer detected routine implementation signals that should be separated from eligible experimental work."
        )
    else:
        body.append(
            "No strong routine implementation signals were detected, but the standard-practice baseline should still be documented."
        )

    gaps = [
        "Identify which standard approaches were tried, why they were insufficient, and what limitation remained unresolved."
    ]

    return {
        "heading": "Standard Practice Baseline",
        "body": body,
        "bullets": routine_signals,
        "gaps": gaps,
    }


def build_investigation_section(cra_checks, evidence_map, case_data):
    investigation_check = cra_checks["systematic_investigation"]
    experiments = evidence_map.get("possible_experiments", [])

    body = [
        "Summarize the systematic investigation through experiments, analysis, prototypes, benchmarks, or iterations.",
        f"Current analyzer finding: {investigation_check['comment']}",
    ]

    gaps = []
    if investigation_check["status"] != "present":
        gaps.append(
            "Add a timeline of hypotheses, tests, iterations, failed attempts, and decision points."
        )

    answer_bullets = build_answer_bullets(case_data.get("answers", []))

    return {
        "heading": "Systematic Investigation",
        "body": body,
        "bullets": experiments + answer_bullets,
        "gaps": gaps,
    }


def build_results_section(cra_checks, evidence_map):
    results_check = cra_checks["experimental_results"]

    body = [
        "Working results summary:",
        evidence_map.get("possible_results", "No results summary available."),
        f"Current analyzer finding: {results_check['comment']}",
    ]

    gaps = []
    if results_check["status"] != "present":
        gaps.append(
            "Document measured results for each approach, including failures and abandoned approaches."
        )

    return {
        "heading": "Results And Technical Learning",
        "body": body,
        "bullets": [],
        "gaps": gaps,
    }


def build_advancement_section(cra_checks):
    advancement_check = cra_checks["technological_advancement"]

    body = [
        "Explain the technological knowledge gained from the investigation, not just the business outcome or finished feature.",
        f"Current analyzer finding: {advancement_check['comment']}",
    ]

    gaps = []
    if advancement_check["status"] != "present":
        gaps.append(
            "Connect the test results to a technological advancement or learning that was not available at the start."
        )

    return {
        "heading": "Technological Advancement",
        "body": body,
        "bullets": [],
        "gaps": gaps,
    }


def build_evidence_section(cra_checks, agent_assessment, evidence_map):
    evidence_check = cra_checks["supporting_evidence"]
    evidence_requests = agent_assessment.get("evidence_requests", [])
    missing_evidence = evidence_map.get("missing_evidence", [])

    body = [
        f"Current analyzer finding: {evidence_check['comment']}",
        "Collect evidence that ties the uncertainty, experiments, results, and learning to contemporaneous records.",
    ]

    gaps = []
    if evidence_check["status"] != "present":
        gaps.append(
            "Attach or reference supporting records before treating the technical report as claim-ready."
        )

    return {
        "heading": "Supporting Evidence Inventory",
        "body": body,
        "bullets": unique_items(evidence_requests + missing_evidence),
        "gaps": gaps,
    }


def build_gaps_section(agent_assessment):
    return {
        "heading": "Open Gaps And Analyst Questions",
        "body": [
            "Use these items to drive the next interview or evidence request cycle."
        ],
        "bullets": unique_items(
            agent_assessment.get("blockers", [])
            + agent_assessment.get("interview_focus", [])
            + agent_assessment.get("action_plan", [])
        ),
        "gaps": [],
    }


def assess_report_readiness(final_assessment):
    checks = final_assessment["cra_checks"]
    statuses = [result["status"] for result in checks.values()]
    present_count = statuses.count("present")
    partial_count = statuses.count("partial")
    missing_count = statuses.count("missing")
    max_score = max(len(statuses) * 2, 1)
    score = round(((present_count * 2) + partial_count) / max_score * 100)
    prediction = final_assessment.get("prediction", "")
    agent_assessment = final_assessment.get("agent_assessment", {})
    case_stage = agent_assessment.get("case_stage", "")

    if prediction == "routine" and case_stage == "routine_screening":
        level = "not_report_ready"
        reasons = ["The work currently appears routine rather than SR&ED candidate work."]
    elif score >= 80 and missing_count == 0:
        level = "draft_ready"
        reasons = ["Most technical report elements are present or partially supported."]
    elif score >= 50:
        level = "draft_with_gaps"
        reasons = ["A report draft can be prepared, but key gaps need analyst follow-up."]
    else:
        level = "intake_required"
        reasons = ["The case record is not detailed enough for a reliable technical report draft."]

    gaps = [
        f"{check_name}: {result['comment']}"
        for check_name, result in checks.items()
        if result["status"] != "present"
    ]

    return {
        "level": level,
        "score": score,
        "present_count": present_count,
        "partial_count": partial_count,
        "missing_count": missing_count,
        "reasons": reasons,
        "gaps": gaps,
    }


def apply_t661_evidence_gate(readiness, evidence_assessment):
    if evidence_assessment["can_draft"]:
        return readiness

    gated = dict(readiness)
    gated["level"] = "intake_required"
    gated["reasons"] = [
        "T661 drafting is withheld until the line-specific evidence gaps are resolved."
    ]
    evidence_gaps = [
        f"Line {line_number}: {missing_item}"
        for line_number, section in evidence_assessment["sections"].items()
        for missing_item in section["missing_information"]
    ]
    gated["gaps"] = unique_items(readiness.get("gaps", []) + evidence_gaps)
    return gated


def render_technical_report(report):
    lines = [
        f"# {report['title']}",
        "",
        f"Case ID: {report['case_id']}",
        f"Case created: {report['created_at']}",
        f"Case status: {report['status']}",
    ]

    if report["source_case_path"]:
        lines.append(f"Source case file: {report['source_case_path']}")

    lines.extend([
        "",
        "## Report Metadata",
    ])

    for label, value in report["metadata"].items():
        if value:
            lines.append(f"- {titleize(label)}: {value}")

    lines.extend([
        f"- Readiness: {report['readiness']['level']}",
        f"- Readiness score: {report['readiness']['score']}%",
        "",
    ])

    append_strategy_lines(lines, report["report_strategy"])
    append_t661_evidence_assessment(lines, report["t661_evidence_assessment"])

    if report["t661_evidence_assessment"]["can_draft"]:
        lines.extend([
            "## T661 Project Description Draft",
            "",
            (
                "These drafts are structured for Form T661 Part 2, Section B. "
                "Review the wording, confirm technical accuracy, and enter the final "
                "version into approved tax software or the form workflow."
            ),
            "",
        ])

        for line_number in ["242", "244", "246"]:
            line = report["t661_project_description"][line_number]
            lines.append(
                f"### Line {line_number} - {line['title']} "
                f"(Maximum {line['word_limit']} words)"
            )
            lines.append("")
            lines.append(f"Word count: {line['word_count']} / {line['word_limit']}")
            lines.append("")
            lines.append(line["draft"])
            lines.append("")

            if line["warnings"]:
                lines.append("Warnings:")
                for warning in line["warnings"]:
                    lines.append(f"- {warning}")
                lines.append("")
    else:
        lines.extend([
            "## T661 Drafting Decision",
            "",
            "**Draft not generated.** The available information is not sufficient for a "
            "grounded response to every required T661 project-description line.",
            "",
            "Complete the evidence questions above, then run the analyzer again.",
            "",
        ])

    lines.extend([
        "## Supporting Analyst Notes",
        "",
    ])

    for section in report["sections"]:
        lines.append(f"### {section['heading']}")
        lines.append("")

        for paragraph in section.get("body", []):
            if paragraph:
                lines.append(paragraph)
                lines.append("")

        bullets = section.get("bullets", [])
        if bullets:
            for bullet in bullets:
                lines.append(f"- {bullet}")
            lines.append("")

        gaps = section.get("gaps", [])
        if gaps:
            lines.append("Gaps to close:")
            for gap in gaps:
                lines.append(f"- {gap}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def append_t661_evidence_assessment(lines, assessment):
    lines.extend([
        "## T661 Evidence Assessment",
        "",
        f"Decision: `{assessment['decision']}`",
        "",
        assessment["summary"],
        "",
    ])

    for line_number in ("242", "244", "246"):
        section = assessment["sections"][line_number]
        lines.extend([
            f"### Line {line_number}: {section['title']}",
            "",
            f"Status: `{section['status']}`",
            "",
            "#### Information supported by the source",
            "",
        ])

        if section["supported_information"]:
            lines.extend(f"- {item}" for item in section["supported_information"])
        else:
            lines.append("- No sufficiently specific information identified.")

        lines.extend(["", "#### Missing information", ""])
        if section["missing_information"]:
            lines.extend(f"- {item}" for item in section["missing_information"])
        else:
            lines.append("- No blocking omissions detected by the local evidence checks.")

        for stream in section["stream_assessments"]:
            lines.extend([
                "",
                f"#### {stream['stream_id']}/{stream['sis_id']}: {stream['title']}",
                "",
                "Supported facts for this stream:",
            ])
            if stream["supported_information"]:
                lines.extend(
                    f"- {item}"
                    for item in stream["supported_information"]
                )
            else:
                lines.append("- No stream-specific source statement identified.")

            if stream["questions"]:
                lines.extend(["", "Questions to resolve:"])
                lines.extend(f"- {question}" for question in stream["questions"])

        lines.append("")


def append_strategy_lines(lines, strategy):
    lines.extend([
        "## Drafting Strategy And Rationale",
        "",
        f"Recommended structure: {strategy['selected_structure']['label']}",
        "",
    ])

    for reason in strategy["rationale"]:
        lines.append(f"- {reason}")
    lines.append("")

    if strategy["streams"]:
        lines.extend([
            "### Candidate TU/SIS Streams",
            "",
        ])

        for stream in strategy["streams"]:
            lines.append(f"#### {stream['id']}: {stream['title']}")
            lines.append("")
            lines.append(f"Uncertainty: {stream['uncertainty']}")
            lines.append("")
            lines.append(f"Systematic investigation: {stream['investigation']}")
            lines.append("")

            if stream["evidence_terms"]:
                lines.append("Detected source terms:")
                for term in stream["evidence_terms"]:
                    lines.append(f"- {term}")
                lines.append("")

            if stream["source_questions"]:
                lines.append("Relevant source sections:")
                for source_question in stream["source_questions"]:
                    lines.append(f"- {source_question}")
                lines.append("")

    if strategy["specific_questions"]:
        lines.extend([
            "### Specific Follow-Up Questions",
            "",
        ])

        for question in strategy["specific_questions"]:
            lines.append(f"- {question}")
        lines.append("")


def extract_questionnaire_sections(text):
    sections = {}
    current_number = None
    current_title = ""
    current_lines = []

    for line in text.splitlines():
        match = QUESTION_HEADING_PATTERN.match(line.strip())
        if match:
            if current_number is not None:
                sections[current_number] = {
                    "title": current_title,
                    "text": "\n".join(current_lines).strip(),
                }

            current_number = int(match.group(1))
            current_title = match.group(2).strip()
            current_lines = []
        elif current_number is not None:
            current_lines.append(line)

    if current_number is not None:
        sections[current_number] = {
            "title": current_title,
            "text": "\n".join(current_lines).strip(),
        }

    return sections


def get_relevant_paragraphs(
    sections,
    section_numbers,
    keywords,
    max_items,
    max_per_section=3,
    fallback_to_section=True,
):
    items = []

    for number in section_numbers:
        section = sections.get(number)
        if not section:
            continue

        candidates = extract_relevant_sentences(section["text"], keywords)
        if not candidates and fallback_to_section:
            candidates = split_paragraphs(section["text"])[:2]

        section_items = 0
        for candidate in candidates:
            cleaned = clean_markdown(candidate)
            if cleaned and not cleaned.endswith(":") and cleaned not in items:
                items.append(cleaned)
                section_items += 1

            if len(items) >= max_items:
                return items

            if section_items >= max_per_section:
                break

    return items


def extract_relevant_sentences(text, keywords):
    sentences = split_sentences(text)
    matches = []
    lowered_keywords = [keyword.lower() for keyword in keywords]

    for sentence in sentences:
        normalized = sentence.lower()
        if any(keyword in normalized for keyword in lowered_keywords):
            matches.append(sentence)

    return matches


def split_sentences(text):
    paragraphs = split_paragraphs(text)
    sentences = []

    for paragraph in paragraphs:
        if paragraph.startswith("- "):
            sentences.append(paragraph[2:].strip())
            continue

        sentences.extend(SENTENCE_PATTERN.split(paragraph))

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_paragraphs(text):
    paragraphs = []
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            continue

        if stripped.startswith("#"):
            continue

        current_lines.append(stripped)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return paragraphs


def limit_words(text, limit):
    words = text.split()

    if len(words) <= limit:
        return text

    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def word_count(text):
    return len(text.split())


def clean_whitespace(text):
    return " ".join(text.split())


def clean_t661_text(text):
    cleaned_lines = []
    previous_blank = False

    for line in text.splitlines():
        cleaned_line = clean_whitespace(line)
        if not cleaned_line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(cleaned_line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def clean_markdown(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`(.*?)`", r"\1", cleaned)
    cleaned = cleaned.replace("###", "").strip()
    return clean_whitespace(cleaned)


def build_answer_bullets(answers):
    bullets = []

    for item in answers:
        answer = item.get("answer", "").strip()
        question = item.get("question", "").strip()
        if answer:
            if question:
                bullets.append(f"{question} Answer: {answer}")
            else:
                bullets.append(answer)

    return bullets


def blockquote(text):
    stripped_text = text.strip()

    if not stripped_text:
        return "> No project description has been captured yet."

    return "\n".join(f"> {line}" if line else ">" for line in stripped_text.splitlines())


def format_confidence(probability_map):
    if not probability_map:
        return ""

    return f"{max(probability_map.values()):.2f}"


def titleize(value):
    return value.replace("_", " ").title()


def sanitize_filename(value):
    cleaned = [
        character
        for character in str(value)
        if character.isalnum() or character in {"-", "_"}
    ]
    return "".join(cleaned) or "case"


def unique_items(items):
    seen = set()
    unique = []

    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate an SR&ED technical report draft from a saved case file.",
    )
    parser.add_argument("case_id", help="Case ID, file name, or JSON path.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the generated report after saving it.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    report_path = generate_technical_report_for_case(args.case_id)

    print("Technical report saved:")
    print(report_path)

    if args.show:
        print()
        print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

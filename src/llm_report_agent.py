import argparse
import getpass
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from analysis_engine import BASE_DIR, analyze_text, load_classifier
from case_store import make_json_safe, summarize_analysis
from evidence_agent import (
    LINE_REQUIREMENTS,
    apply_evidence_audit,
    assess_evidence_graph,
    build_stream_summaries,
    evidence_item_supports_requirement,
    request_evidence_graph,
    request_evidence_graph_audit,
    validate_evidence_audit_payload,
)
from grounding import find_unsupported_numeric_facts
from report_strategy import build_report_strategy
from responses_http_client import ResponsesHTTPClient
from t661_evidence_assessment import build_t661_evidence_assessment
from technical_report import (
    T661_LINE_WORD_LIMITS,
    T661_LINE_TITLES,
    build_t661_line,
    build_t661_project_description,
    extract_questionnaire_sections,
    sanitize_filename,
    word_count,
)


DEFAULT_MODEL = "gpt-5.6-terra"
OUTPUT_DIR = "technical_reports"
ELIGIBILITY_SIGNALS = {
    "strong_candidate",
    "possible_candidate",
    "insufficient_information",
    "likely_routine",
}
STRUCTURE_MODES = {
    "integrated_narrative",
    "split_by_uncertainty_stream",
    "hybrid",
}
CONFIDENCE_LEVELS = {"low", "medium", "high"}
DRAFTING_DECISIONS = {"draft_ready", "needs_more_information"}
REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "eligibility_signal": {
            "type": "string",
            "enum": sorted(ELIGIBILITY_SIGNALS),
        },
        "confidence": {
            "type": "string",
            "enum": sorted(CONFIDENCE_LEVELS),
        },
        "structure_mode": {
            "type": "string",
            "enum": sorted(STRUCTURE_MODES),
        },
        "structure_rationale": {"type": "string"},
        "drafting_decision": {
            "type": "string",
            "enum": sorted(DRAFTING_DECISIONS),
        },
        "section_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_number": {
                        "type": "string",
                        "enum": ["242", "244", "246"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["ready", "needs_more_information"],
                    },
                    "supported_information": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing_information": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "line_number",
                    "status",
                    "supported_information",
                    "missing_information",
                ],
                "additionalProperties": False,
            },
        },
        "technical_streams": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "uncertainty": {"type": "string"},
                    "standard_practice_gap": {"type": "string"},
                    "systematic_investigation": {"type": "string"},
                    "advancement": {"type": "string"},
                    "source_support": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "title",
                    "uncertainty",
                    "standard_practice_gap",
                    "systematic_investigation",
                    "advancement",
                    "source_support",
                    "evidence_gaps",
                ],
                "additionalProperties": False,
            },
        },
        "line_242": {"type": "string"},
        "line_244": {"type": "string"},
        "line_246": {"type": "string"},
        "follow_up_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "examples_to_check": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "question",
                    "why_it_matters",
                    "examples_to_check",
                ],
                "additionalProperties": False,
            },
        },
        "factual_risks": {
            "type": "array",
            "items": {"type": "string"},
        },
        "review_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "overall_assessment",
        "eligibility_signal",
        "confidence",
        "structure_mode",
        "structure_rationale",
        "drafting_decision",
        "section_assessments",
        "technical_streams",
        "line_242",
        "line_244",
        "line_246",
        "follow_up_questions",
        "factual_risks",
        "review_notes",
    ],
    "additionalProperties": False,
}


DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "structure_mode": {
            "type": "string",
            "enum": sorted(STRUCTURE_MODES),
        },
        "structure_rationale": {"type": "string"},
        "line_242": {"type": "string"},
        "line_244": {"type": "string"},
        "line_246": {"type": "string"},
        "draft_support": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_number": {
                        "type": "string",
                        "enum": ["242", "244", "246"],
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "coverage_note": {"type": "string"},
                },
                "required": ["line_number", "evidence_ids", "coverage_note"],
                "additionalProperties": False,
            },
        },
        "claim_support": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "line_number": {
                        "type": "string",
                        "enum": ["242", "244", "246"],
                    },
                    "claim_text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "claim_id",
                    "line_number",
                    "claim_text",
                    "evidence_ids",
                ],
                "additionalProperties": False,
            },
        },
        "factual_risks": {
            "type": "array",
            "items": {"type": "string"},
        },
        "review_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "overall_assessment",
        "structure_mode",
        "structure_rationale",
        "line_242",
        "line_244",
        "line_246",
        "draft_support",
        "claim_support",
        "factual_risks",
        "review_notes",
    ],
    "additionalProperties": False,
}


GROUNDING_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "ambiguous", "unsupported"],
                    },
                    "reason": {"type": "string"},
                    "evidence_ids_reviewed": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "claim_id",
                    "verdict",
                    "reason",
                    "evidence_ids_reviewed",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overall_assessment", "claims"],
    "additionalProperties": False,
}


DRAFTING_INSTRUCTIONS = """
You are the drafting stage of a Canadian SR&ED technical-report system. The supplied
evidence graph has already passed a local completeness gate. Draft Form T661 project
description Lines 242, 244, and 246 for human technical and tax review.

Use only accepted evidence items in the graph. Do not use the original source as an
independent basis for new facts, and do not turn inferred, ambiguous, negated,
prior-year, future, vendor, or routine work into claimant facts. Do not invent or
calculate measurements, dates, tests, records, results, causal explanations, or
conclusions. Stream titles are organizational labels, not evidence. Preserve material
qualifications and unsuccessful results.

Choose an integrated, split-stream, or hybrid structure based on the actual evidence.
When separate uncertainties have different hypotheses or investigations, use TU1,
TU2, SIS1, SIS2, and corresponding advancement labels inside the relevant line.
Avoid duplicating common facts across streams.

Use the validated investigation_sequences as the causal backbone for Lines 244 and
246. Do not combine a hypothesis from one sequence with the work, result, conclusion,
or advancement from another. When several sequences represent alternatives, refinements,
or pivots, preserve that progression and identify the result that led to each decision.

Line 242 must explain the technological objective, starting knowledge, limits of
standard practice, and technological uncertainty. Line 244 must describe claimed-year
hypotheses, experiments or analysis, observations, and conclusions in a coherent
sequence. Line 246 must state technological knowledge gained or attempted, including
useful learning from failed work, rather than product or business benefits.

Keep Lines 242 and 246 at or below 350 words and Line 244 at or below 700 words. For
each line, list every evidence ID used. The cited IDs must collectively cover all
required evidence categories for that line. Also return one claim_support entry for
every sentence or standalone factual statement in the three drafts. claim_text must be
an exact substring of its draft line, and its evidence IDs must support every factual
detail in that statement. Do not state that the project is legally eligible and do not
add boilerplate claims about CRA compliance.
""".strip()


GROUNDING_AUDIT_INSTRUCTIONS = """
You are the independent grounding-audit stage of a Canadian SR&ED report system.
Review every drafted claim against only the accepted evidence items cited for that
claim. Do not rely on the draft's support explanation, general plausibility, project
summary, stream title, or outside knowledge.

Mark a claim supported only when its cited exact source quotes and normalized facts
support every material detail without inference. Mark it ambiguous when the evidence
is related but does not establish the full claim. Mark it unsupported when it adds or
changes a fact, causal relationship, chronology, attribution, result, record, number,
or conclusion. Preserve failed and qualified results. Audit every claim ID exactly
once and list the evidence IDs actually reviewed.
""".strip()


def build_local_context(source_text, classifier=None):
    classifier = classifier or load_classifier()
    analysis = analyze_text(source_text, classifier)
    final_assessment = summarize_analysis(analysis)
    source_sections = extract_questionnaire_sections(source_text)
    case_data = {
        "case_id": "local_capability_test",
        "original_text": source_text,
        "updated_text": source_text,
        "final_assessment": final_assessment,
    }
    strategy = build_report_strategy(
        case_data,
        final_assessment,
        source_sections,
    )
    evidence_assessment = build_t661_evidence_assessment(
        source_text,
        source_sections,
        strategy,
    )
    deterministic_t661 = build_t661_project_description(
        case_data,
        final_assessment,
        source_sections,
        strategy,
        evidence_assessment,
    )

    return make_json_safe({
        "project_source": source_text,
        "local_analysis": final_assessment,
        "local_strategy": strategy,
        "t661_evidence_assessment": evidence_assessment,
        "local_t661_baseline": deterministic_t661,
    })


def build_model_input(context):
    analyzer_context = {
        key: value
        for key, value in context.items()
        if key != "project_source"
    }
    return (
        "PROJECT SOURCE\n"
        "<project_source>\n"
        f"{context['project_source'].strip()}\n"
        "</project_source>\n\n"
        "LOCAL ANALYZER CONTEXT\n"
        f"{json.dumps(analyzer_context, indent=2, sort_keys=True)}"
    )


def build_grounded_draft_input(context, graph, readiness):
    drafting_graph = {
        "claimed_tax_year": graph["claimed_tax_year"],
        "claimed_tax_year_source_quote": graph["claimed_tax_year_source_quote"],
        "technical_streams": [
            {"id": stream["id"], "title": stream["title"]}
            for stream in graph["technical_streams"]
        ],
        "evidence_items": graph["evidence_items"],
        "investigation_sequences": graph["investigation_sequences"],
    }
    return (
        "VALIDATED EVIDENCE GRAPH\n"
        f"{json.dumps(drafting_graph, indent=2, sort_keys=True)}\n\n"
        "LOCAL READINESS DECISION\n"
        f"{json.dumps(readiness, indent=2, sort_keys=True)}\n\n"
        "SOURCE USE RULE\n"
        "Draft only from accepted evidence items above. The original source is retained "
        "locally for post-draft verification and is not a license to add uncited facts."
    )


def request_grounded_draft(
    context,
    graph,
    readiness,
    client,
    model,
    reasoning_effort="high",
):
    request = {
        "model": model,
        "instructions": DRAFTING_INSTRUCTIONS,
        "input": build_grounded_draft_input(context, graph, readiness),
        "max_output_tokens": 9000,
        "store": False,
        "prompt_cache_key": "sred-grounded-drafting-v1",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sred_grounded_draft",
                "strict": True,
                "schema": DRAFT_SCHEMA,
            }
        },
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request)
    if not response.output_text:
        raise RuntimeError("The grounded drafting stage returned no output.")

    try:
        payload = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The grounded drafting stage returned invalid JSON.") from exc

    return payload, extract_usage(response)


def build_grounding_audit_input(graph, draft_payload):
    evidence_index = {
        item["id"]: item
        for item in graph["evidence_items"]
    }
    cited_ids = unique_text_items(
        evidence_id
        for claim in draft_payload["claim_support"]
        for evidence_id in claim["evidence_ids"]
    )
    audit_evidence = [
        evidence_index[evidence_id]
        for evidence_id in cited_ids
        if evidence_id in evidence_index
    ]
    audit_payload = {
        "drafts": {
            line_number: draft_payload[f"line_{line_number}"]
            for line_number in ("242", "244", "246")
        },
        "claim_support": draft_payload["claim_support"],
        "cited_evidence_items": audit_evidence,
    }
    return json.dumps(audit_payload, indent=2, sort_keys=True)


def request_grounding_audit(
    graph,
    draft_payload,
    client,
    model,
    reasoning_effort="high",
):
    request = {
        "model": model,
        "instructions": GROUNDING_AUDIT_INSTRUCTIONS,
        "input": build_grounding_audit_input(graph, draft_payload),
        "max_output_tokens": 6000,
        "store": False,
        "prompt_cache_key": "sred-grounding-audit-v1",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sred_grounding_audit",
                "strict": True,
                "schema": GROUNDING_AUDIT_SCHEMA,
            }
        },
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request)
    if not response.output_text:
        raise RuntimeError("The grounding-audit stage returned no output.")

    try:
        payload = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The grounding-audit stage returned invalid JSON.") from exc

    return payload, extract_usage(response)


def request_llm_report(
    context,
    model=DEFAULT_MODEL,
    api_key=None,
    reasoning_effort="high",
    client=None,
):
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            "Reasoning effort must be one of: " + ", ".join(sorted(REASONING_EFFORTS))
        )

    if client is None:
        client = build_responses_client(resolve_api_key(api_key))

    source_text = context["project_source"]
    advisory_context = {
        "local_analysis": context.get("local_analysis", {}),
        "local_strategy": context.get("local_strategy", {}),
        "local_evidence_assessment": context.get("t661_evidence_assessment", {}),
    }
    graph, extraction_usage = request_evidence_graph(
        source_text,
        client,
        model,
        local_context=advisory_context,
        reasoning_effort=reasoning_effort,
    )
    evidence_audit, evidence_audit_usage = request_evidence_graph_audit(
        source_text,
        graph,
        client,
        model,
        reasoning_effort=reasoning_effort,
    )
    pre_draft_usage = combine_usage(extraction_usage, evidence_audit_usage)
    try:
        validate_evidence_audit_payload(evidence_audit, graph)
        graph = apply_evidence_audit(graph, evidence_audit)
    except ValueError as exc:
        readiness = assess_evidence_graph(graph)
        readiness = apply_global_hard_blocker(
            readiness,
            "Independent evidence audit failed: " + str(exc),
        )
        readiness = apply_local_hard_blockers(
            readiness,
            context.get("t661_evidence_assessment", {}),
        )
        report = build_evidence_agent_report(
            graph,
            readiness,
            evidence_audit=evidence_audit,
            evidence_audit_failure=str(exc),
        )
        return report, pre_draft_usage

    readiness = assess_evidence_graph(graph)
    readiness = apply_local_hard_blockers(
        readiness,
        context.get("t661_evidence_assessment", {}),
    )

    if not readiness["can_draft"]:
        report = build_evidence_agent_report(
            graph,
            readiness,
            evidence_audit=evidence_audit,
        )
        return report, pre_draft_usage

    draft_payload, drafting_usage = request_grounded_draft(
        context,
        graph,
        readiness,
        client,
        model,
        reasoning_effort=reasoning_effort,
    )
    try:
        normalize_grounded_draft(
            draft_payload,
            source_text,
            graph,
            readiness,
        )
    except ValueError as exc:
        report = build_evidence_agent_report(
            graph,
            readiness,
            evidence_audit=evidence_audit,
            audit_failure=str(exc),
            audit_failure_stage="post_draft_local_audit",
        )
        return report, combine_usage(pre_draft_usage, drafting_usage)

    grounding_audit, audit_usage = request_grounding_audit(
        graph,
        draft_payload,
        client,
        model,
        reasoning_effort=reasoning_effort,
    )
    usage = combine_usage(pre_draft_usage, drafting_usage, audit_usage)
    try:
        validate_grounding_audit_payload(grounding_audit, draft_payload)
        report = build_evidence_agent_report(
            graph,
            readiness,
            draft_payload=draft_payload,
            evidence_audit=evidence_audit,
            grounding_audit=grounding_audit,
        )
    except ValueError as exc:
        report = build_evidence_agent_report(
            graph,
            readiness,
            evidence_audit=evidence_audit,
            grounding_audit=grounding_audit,
            audit_failure=str(exc),
            audit_failure_stage="independent_grounding_audit",
        )
    return report, usage


def build_responses_client(api_key):
    try:
        from openai import OpenAI
    except ImportError:
        return ResponsesHTTPClient(api_key=api_key)
    sdk_client = OpenAI(api_key=api_key)
    if not hasattr(sdk_client, "responses"):
        return ResponsesHTTPClient(api_key=api_key)
    return sdk_client


def apply_local_hard_blockers(readiness, local_assessment):
    sections = {
        line_number: {
            **section,
            "missing_information": list(section["missing_information"]),
        }
        for line_number, section in readiness["sections"].items()
    }
    hard_blockers = list(readiness.get("hard_blockers", []))

    routine_flags = local_assessment.get("routine_flags", [])
    if routine_flags:
        message = (
            "Local safeguard identified routine vendor or standard implementation: "
            + "; ".join(str(flag) for flag in routine_flags)
        )
        hard_blockers.append(message)
        for section in sections.values():
            section["status"] = "needs_more_information"
            section["missing_information"] = unique_text_items(
                [*section["missing_information"], message]
            )

    for issue in local_assessment.get("consistency_issues", []):
        message = str(issue.get("message", "")).strip()
        if not message:
            continue
        hard_blockers.append(message)
        for line_number in issue.get("lines", []):
            if line_number not in sections:
                continue
            sections[line_number]["status"] = "needs_more_information"
            sections[line_number]["missing_information"] = unique_text_items(
                [*sections[line_number]["missing_information"], message]
            )

    blocked_lines = [
        line_number
        for line_number, section in sections.items()
        if section["status"] != "ready"
    ]
    return {
        **readiness,
        "decision": "draft_ready" if not blocked_lines else "needs_more_information",
        "can_draft": not blocked_lines and bool(sections),
        "blocked_lines": blocked_lines,
        "sections": sections,
        "hard_blockers": unique_text_items(hard_blockers),
    }


def apply_global_hard_blocker(readiness, message):
    sections = {
        line_number: {
            **section,
            "status": "needs_more_information",
            "missing_information": unique_text_items(
                [*section["missing_information"], message]
            ),
        }
        for line_number, section in readiness["sections"].items()
    }
    return {
        **readiness,
        "decision": "needs_more_information",
        "can_draft": False,
        "blocked_lines": list(sections),
        "sections": sections,
        "hard_blockers": unique_text_items(
            [*readiness.get("hard_blockers", []), message]
        ),
    }


def build_evidence_agent_report(
    graph,
    readiness,
    draft_payload=None,
    evidence_audit=None,
    evidence_audit_failure=None,
    grounding_audit=None,
    audit_failure=None,
    audit_failure_stage=None,
):
    evidence_audit_complete = graph["validation"].get("semantic_audit_passed") is True
    grounding_audit_complete = False
    if draft_payload is not None and grounding_audit is not None:
        try:
            validate_grounding_audit_payload(grounding_audit, draft_payload)
        except ValueError:
            grounding_audit_complete = False
        else:
            grounding_audit_complete = True
    draft_ready = (
        draft_payload is not None
        and readiness["can_draft"]
        and evidence_audit_complete
        and grounding_audit_complete
        and evidence_audit_failure is None
        and audit_failure is None
    )
    routine_signal = readiness.get("routine_only") or any(
        "routine" in blocker.lower()
        for blocker in readiness.get("hard_blockers", [])
    )
    accepted_count = graph["validation"]["accepted_evidence_items"]
    rejected_count = graph["validation"]["rejected_evidence_items"]

    if evidence_audit_failure:
        overall_assessment = (
            "The extracted evidence could not pass the independent semantic audit, so "
            f"readiness and T661 drafting were withheld: {evidence_audit_failure}"
        )
        structure_mode = (
            "split_by_uncertainty_stream"
            if len(graph["technical_streams"]) > 1
            else "integrated_narrative"
        )
        structure_rationale = (
            "No report structure is released until every extracted evidence item is "
            "accounted for by the semantic audit."
        )
    elif audit_failure:
        overall_assessment = (
            "The validated evidence is complete enough for grounded drafting, but the "
            f"generated draft was withheld by the post-draft audit: {audit_failure}"
        )
        structure_mode = (
            draft_payload["structure_mode"]
            if draft_payload
            else "split_by_uncertainty_stream"
            if len(graph["technical_streams"]) > 1
            else "integrated_narrative"
        )
        structure_rationale = (
            draft_payload["structure_rationale"]
            if draft_payload
            else "The generated draft was withheld before its proposed structure could be released."
        )
    elif draft_payload:
        overall_assessment = draft_payload["overall_assessment"]
        structure_mode = draft_payload["structure_mode"]
        structure_rationale = draft_payload["structure_rationale"]
    else:
        overall_assessment = build_readiness_summary(readiness)
        structure_mode = (
            "split_by_uncertainty_stream"
            if len(graph["technical_streams"]) > 1
            else "integrated_narrative"
        )
        structure_rationale = (
            "Separate TU/SIS treatment is clearer because the validated graph contains "
            f"{len(graph['technical_streams'])} technical uncertainty streams."
            if len(graph["technical_streams"]) > 1
            else "The validated graph contains one technical uncertainty stream."
        )

    factual_risks = []
    for item in graph["validation"].get("rejected_items", []):
        factual_risks.append(
            "Rejected extracted item "
            f"{item.get('original_id', 'unknown')}: {', '.join(item.get('reasons', []))}."
        )
    for sequence in graph["validation"].get("rejected_sequences", []):
        factual_risks.append(
            "Rejected investigation sequence "
            f"{sequence.get('original_id', 'unknown')}: "
            f"{', '.join(sequence.get('reasons', []))}."
        )
    for issue in graph["validation"].get("rejected_issues", []):
        factual_risks.append(
            "Rejected extracted issue "
            f"{issue.get('original_id', 'unknown')}: "
            f"{', '.join(issue.get('reasons', []))}."
        )
    factual_risks.extend(
        "Contradiction: " + item["description"]
        for item in graph["contradictions"]
    )
    factual_risks.extend(
        "Attribution issue: " + item["description"]
        for item in graph["attribution_issues"]
    )
    factual_risks.extend(readiness.get("hard_blockers", []))
    if evidence_audit_failure:
        factual_risks.insert(0, "Evidence audit failure: " + evidence_audit_failure)
    if audit_failure:
        factual_risks.insert(0, "Draft audit failure: " + audit_failure)
    if draft_payload:
        factual_risks.extend(draft_payload["factual_risks"])

    review_notes = [
        "Every accepted evidence item was matched to an exact contiguous source quote.",
        "Human technical and tax review remains required; this is not an eligibility opinion.",
    ]
    if evidence_audit and not evidence_audit_failure:
        review_notes.insert(
            1,
            "The claimed tax year, every accepted evidence item, and each investigation "
            "sequence passed an independent semantic audit.",
        )
    if draft_payload:
        review_notes.extend(draft_payload["review_notes"])

    line_values = {
        line_number: draft_payload[f"line_{line_number}"] if draft_ready else ""
        for line_number in ("242", "244", "246")
    }
    report = {
        "overall_assessment": overall_assessment,
        "eligibility_signal": (
            "likely_routine"
            if routine_signal
            else "possible_candidate"
            if readiness["can_draft"]
            else "insufficient_information"
        ),
        "confidence": (
            "high"
            if accepted_count >= 10 and rejected_count == 0
            else "medium"
            if accepted_count >= 4
            else "low"
        ),
        "structure_mode": structure_mode,
        "structure_rationale": structure_rationale,
        "drafting_decision": "draft_ready" if draft_ready else "needs_more_information",
        "section_assessments": build_section_assessments(readiness),
        "technical_streams": build_stream_summaries(graph, readiness),
        **{f"line_{line_number}": value for line_number, value in line_values.items()},
        "follow_up_questions": normalize_follow_up_questions(
            readiness["follow_up_questions"]
        ),
        "factual_risks": unique_text_items(factual_risks),
        "review_notes": unique_text_items(review_notes),
        "t661_lines": {},
        "evidence_graph": graph,
        "evidence_audit": evidence_audit or graph.get("evidence_audit", {}),
        "draft_support": draft_payload["draft_support"] if draft_ready else [],
        "claim_support": draft_payload["claim_support"] if draft_ready else [],
        "grounding_audit": grounding_audit or {},
        "agent_stages": [
            {"stage": "evidence_extraction", "status": "complete"},
            {
                "stage": "independent_evidence_audit",
                "status": (
                    "failed"
                    if evidence_audit_failure
                    else "passed"
                    if evidence_audit or graph.get("evidence_audit")
                    else "not_run"
                ),
            },
            {
                "stage": "readiness_gate",
                "status": "passed" if readiness["can_draft"] else "blocked",
            },
            {
                "stage": "grounded_drafting",
                "status": "complete" if readiness["can_draft"] else "not_run",
            },
            {
                "stage": "post_draft_local_audit",
                "status": (
                    "failed"
                    if audit_failure_stage == "post_draft_local_audit"
                    else "passed"
                    if draft_payload or grounding_audit
                    else "not_run"
                ),
            },
            {
                "stage": "independent_grounding_audit",
                "status": (
                    "failed"
                    if audit_failure_stage == "independent_grounding_audit"
                    else "passed"
                    if grounding_audit and draft_ready
                    else "not_run"
                ),
            },
        ],
        "draft_audit": {
            "status": "failed" if audit_failure else "passed" if draft_ready else "not_run",
            "message": audit_failure or "",
            "failed_stage": audit_failure_stage or "",
        },
        "evidence_audit_status": {
            "status": (
                "failed"
                if evidence_audit_failure
                else "passed"
                if evidence_audit or graph.get("evidence_audit")
                else "not_run"
            ),
            "message": evidence_audit_failure or "",
        },
    }
    if draft_ready:
        report["t661_lines"] = {
            line_number: build_t661_line(line_number, line_values[line_number])
            for line_number in ("242", "244", "246")
        }
    return report


def normalize_grounded_draft(payload, source_text, graph, readiness):
    validate_grounded_draft_payload(payload)
    evidence_index = {item["id"]: item for item in graph["evidence_items"]}
    support_by_line = {
        item["line_number"]: item
        for item in payload["draft_support"]
    }
    if len(payload["draft_support"]) != 3 or set(support_by_line) != {"242", "244", "246"}:
        raise ValueError("The draft must provide one support map for each T661 line.")

    validate_claim_support(payload, support_by_line, evidence_index)

    for line_number in ("242", "244", "246"):
        draft = str(payload[f"line_{line_number}"]).strip()
        if not draft:
            raise ValueError(f"Line {line_number} was empty.")
        if word_count(draft) > T661_LINE_WORD_LIMITS[line_number]:
            raise ValueError(
                f"Line {line_number} exceeded its {T661_LINE_WORD_LIMITS[line_number]}-word limit."
            )

        evidence_ids = support_by_line[line_number]["evidence_ids"]
        if not evidence_ids:
            raise ValueError(f"Line {line_number} cited no evidence IDs.")
        unknown_ids = sorted(set(evidence_ids) - set(evidence_index))
        if unknown_ids:
            raise ValueError(
                f"Line {line_number} cited unknown evidence IDs: {', '.join(unknown_ids)}."
            )
        validate_line_support(
            line_number,
            evidence_ids,
            evidence_index,
            graph["technical_streams"],
            graph["investigation_sequences"],
        )

    generated_text = "\n".join(
        payload[f"line_{line_number}"]
        for line_number in ("242", "244", "246")
    )
    unsupported_numbers = find_unsupported_numeric_facts(source_text, generated_text)
    if unsupported_numbers:
        raise ValueError(
            "The draft introduced numeric content not found in the project source: "
            + ", ".join(unsupported_numbers)
            + "."
        )

    return payload


def validate_grounded_draft_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("The grounded draft must be a JSON object.")
    missing = [key for key in DRAFT_SCHEMA["required"] if key not in payload]
    if missing:
        raise ValueError("The grounded draft is missing fields: " + ", ".join(missing))
    if payload["structure_mode"] not in STRUCTURE_MODES:
        raise ValueError("The grounded draft has an invalid structure mode.")
    for field in ("draft_support", "claim_support", "factual_risks", "review_notes"):
        if not isinstance(payload[field], list):
            raise ValueError(f"The grounded draft field '{field}' must be a list.")
    for field in (
        "overall_assessment",
        "structure_rationale",
        "line_242",
        "line_244",
        "line_246",
    ):
        if not isinstance(payload[field], str):
            raise ValueError(f"The grounded draft field '{field}' must be a string.")

    for support in payload["draft_support"]:
        if not isinstance(support, dict):
            raise ValueError("Every draft support entry must be an object.")
        required = {"line_number", "evidence_ids", "coverage_note"}
        if not required.issubset(support):
            raise ValueError("A draft support entry is missing required fields.")
        if support["line_number"] not in {"242", "244", "246"}:
            raise ValueError("A draft support entry has an invalid line number.")
        if not isinstance(support["evidence_ids"], list) or not all(
            isinstance(item, str) for item in support["evidence_ids"]
        ):
            raise ValueError("Draft support evidence IDs must be a list of strings.")
        if not isinstance(support["coverage_note"], str):
            raise ValueError("A draft support coverage note must be a string.")

    seen_claim_ids = set()
    for claim in payload["claim_support"]:
        if not isinstance(claim, dict):
            raise ValueError("Every claim support entry must be an object.")
        required = {"claim_id", "line_number", "claim_text", "evidence_ids"}
        if not required.issubset(claim):
            raise ValueError("A claim support entry is missing required fields.")
        claim_id = claim["claim_id"]
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("Every claim support entry must have a non-empty claim ID.")
        if claim_id in seen_claim_ids:
            raise ValueError(f"Duplicate claim support ID: {claim_id}.")
        seen_claim_ids.add(claim_id)
        if claim["line_number"] not in {"242", "244", "246"}:
            raise ValueError("A claim support entry has an invalid line number.")
        if not isinstance(claim["claim_text"], str) or not claim["claim_text"].strip():
            raise ValueError("Every claim support entry must have non-empty claim text.")
        if not isinstance(claim["evidence_ids"], list) or not claim["evidence_ids"]:
            raise ValueError("Every drafted claim must cite at least one evidence ID.")
        if not all(isinstance(item, str) and item.strip() for item in claim["evidence_ids"]):
            raise ValueError("Claim support evidence IDs must be non-empty strings.")

    for field in ("factual_risks", "review_notes"):
        if not all(isinstance(item, str) for item in payload[field]):
            raise ValueError(f"Every grounded draft {field} item must be a string.")


def split_draft_claims(draft):
    claims = []
    for block in re.split(r"\n+", str(draft)):
        block = block.strip()
        if not block:
            continue
        start = 0
        for match in re.finditer(r"[.!?](?=\s+[A-Z0-9]|$)", block):
            claim = block[start:match.end()].strip()
            if claim:
                claims.append(claim)
            start = match.end()
            while start < len(block) and block[start].isspace():
                start += 1
        remainder = block[start:].strip()
        if remainder:
            claims.append(remainder)
    return claims


def validate_claim_support(payload, support_by_line, evidence_index):
    claims_by_line = {line_number: [] for line_number in ("242", "244", "246")}
    for claim in payload["claim_support"]:
        line_number = claim["line_number"]
        draft = payload[f"line_{line_number}"]
        claim_text = claim["claim_text"].strip()
        if claim_text not in draft:
            raise ValueError(
                f"Claim {claim['claim_id']} is not an exact substring of Line {line_number}."
            )
        unknown_ids = sorted(set(claim["evidence_ids"]) - set(evidence_index))
        if unknown_ids:
            raise ValueError(
                f"Claim {claim['claim_id']} cited unknown evidence IDs: "
                + ", ".join(unknown_ids)
                + "."
            )
        line_ids = set(support_by_line[line_number]["evidence_ids"])
        outside_line = sorted(set(claim["evidence_ids"]) - line_ids)
        if outside_line:
            raise ValueError(
                f"Claim {claim['claim_id']} cited evidence outside Line {line_number}'s "
                f"support map: {', '.join(outside_line)}."
            )
        claims_by_line[line_number].append(claim_text)

    for line_number in ("242", "244", "246"):
        expected_claims = split_draft_claims(payload[f"line_{line_number}"])
        supplied_claims = claims_by_line[line_number]
        missing = [claim for claim in expected_claims if supplied_claims.count(claim) == 0]
        duplicate = [claim for claim in expected_claims if supplied_claims.count(claim) > 1]
        extra = [claim for claim in supplied_claims if claim not in expected_claims]
        if missing or duplicate or extra or len(supplied_claims) != len(expected_claims):
            details = []
            if missing:
                details.append("missing: " + " | ".join(missing))
            if duplicate:
                details.append("duplicated: " + " | ".join(unique_text_items(duplicate)))
            if extra:
                details.append("not a complete sentence/statement: " + " | ".join(extra))
            raise ValueError(
                f"Line {line_number} claim support must map every sentence exactly once"
                + (": " + "; ".join(details) if details else ".")
            )


def validate_grounding_audit_payload(payload, draft_payload):
    if not isinstance(payload, dict):
        raise ValueError("The independent grounding audit must be a JSON object.")
    missing = [key for key in GROUNDING_AUDIT_SCHEMA["required"] if key not in payload]
    if missing:
        raise ValueError("The grounding audit is missing fields: " + ", ".join(missing))
    if not isinstance(payload["overall_assessment"], str):
        raise ValueError("The grounding audit assessment must be a string.")
    if not isinstance(payload["claims"], list):
        raise ValueError("The grounding audit claims field must be a list.")

    expected = {claim["claim_id"]: claim for claim in draft_payload["claim_support"]}
    audited = {}
    for audit in payload["claims"]:
        if not isinstance(audit, dict):
            raise ValueError("Every grounding audit claim must be an object.")
        required = {"claim_id", "verdict", "reason", "evidence_ids_reviewed"}
        if not required.issubset(audit):
            raise ValueError("A grounding audit claim is missing required fields.")
        claim_id = audit["claim_id"]
        if claim_id not in expected:
            raise ValueError(f"The grounding audit returned unknown claim ID {claim_id}.")
        if claim_id in audited:
            raise ValueError(f"The grounding audit duplicated claim ID {claim_id}.")
        if audit["verdict"] not in {"supported", "ambiguous", "unsupported"}:
            raise ValueError(f"The grounding audit returned an invalid verdict for {claim_id}.")
        if not isinstance(audit["reason"], str) or not audit["reason"].strip():
            raise ValueError(f"The grounding audit returned no reason for {claim_id}.")
        reviewed = audit["evidence_ids_reviewed"]
        if not isinstance(reviewed, list) or not reviewed or not all(
            isinstance(item, str) and item.strip() for item in reviewed
        ):
            raise ValueError(f"The grounding audit reviewed no valid evidence for {claim_id}.")
        cited = expected[claim_id]["evidence_ids"]
        if set(reviewed) != set(cited):
            raise ValueError(
                f"The grounding audit did not review exactly the cited evidence for {claim_id}."
            )
        audited[claim_id] = audit

    missing_claims = sorted(set(expected) - set(audited))
    if missing_claims:
        raise ValueError(
            "The grounding audit omitted claim IDs: " + ", ".join(missing_claims) + "."
        )
    failed = [
        f"{claim_id} ({audit['verdict']}): {audit['reason'].strip()}"
        for claim_id, audit in audited.items()
        if audit["verdict"] != "supported"
    ]
    if failed:
        raise ValueError("Independent grounding audit rejected claims: " + "; ".join(failed))


def validate_line_support(
    line_number,
    evidence_ids,
    evidence_index,
    streams,
    investigation_sequences,
):
    cited_items = [evidence_index[evidence_id] for evidence_id in evidence_ids]
    cited_ids = set(evidence_ids)
    for stream in streams:
        for category in LINE_REQUIREMENTS[line_number]:
            supported = any(
                item["category"] == category
                and evidence_item_supports_requirement(item, category)
                and item["stream_id"] in {stream["id"], "GLOBAL"}
                for item in cited_items
            )
            if not supported:
                raise ValueError(
                    f"Line {line_number} support omitted {category} evidence for "
                    f"{stream['id']}."
                )

        stream_sequences = [
            sequence
            for sequence in investigation_sequences
            if sequence["stream_id"] == stream["id"]
        ]
        if line_number == "242":
            linked = any(
                cited_ids.intersection(sequence["uncertainty_evidence_ids"])
                for sequence in stream_sequences
            )
            if not linked:
                raise ValueError(
                    f"Line 242 support did not cite an uncertainty linked to a validated "
                    f"investigation sequence for {stream['id']}."
                )
        elif line_number == "244":
            linked = any(
                set(
                    evidence_id
                    for field in (
                        "hypothesis_evidence_ids",
                        "experiment_evidence_ids",
                        "result_evidence_ids",
                        "conclusion_evidence_ids",
                    )
                    for evidence_id in sequence[field]
                ).issubset(cited_ids)
                for sequence in stream_sequences
            )
            if not linked:
                raise ValueError(
                    f"Line 244 support combined evidence without citing one complete "
                    f"validated investigation sequence for {stream['id']}."
                )
        elif line_number == "246":
            linked = any(
                cited_ids.intersection(sequence["advancement_evidence_ids"])
                for sequence in stream_sequences
            )
            if not linked:
                raise ValueError(
                    f"Line 246 support did not cite advancement evidence linked to a "
                    f"validated investigation sequence for {stream['id']}."
                )


def build_section_assessments(readiness):
    return [
        {
            "line_number": line_number,
            "status": section["status"],
            "supported_information": section["supported_information"],
            "missing_information": section["missing_information"],
        }
        for line_number, section in readiness["sections"].items()
    ]


def normalize_follow_up_questions(questions):
    return [
        {
            "question": item["question"],
            "why_it_matters": item["why_it_matters"],
            "examples_to_check": item["examples_to_check"],
        }
        for item in questions
    ]


def build_readiness_summary(readiness):
    if readiness.get("routine_only"):
        return (
            "The source currently supports routine implementation but does not support a "
            "technological uncertainty requiring systematic investigation."
        )
    blocked = ", ".join(readiness["blocked_lines"])
    return (
        "The evidence graph is not complete enough to draft all T661 project-description "
        f"lines. Resolve the identified gaps for Line(s) {blocked}."
    )


def combine_usage(*usage_records):
    combined = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        values = [record.get(key) for record in usage_records if record.get(key) is not None]
        combined[key] = sum(values) if values else None
    return combined


def unique_text_items(items):
    seen = set()
    result = []
    for item in items:
        text_value = str(item).strip()
        marker = text_value.casefold()
        if marker and marker not in seen:
            result.append(text_value)
            seen.add(marker)
    return result


def resolve_api_key(api_key=None):
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if resolved_key:
        return resolved_key

    if sys.stdin.isatty():
        resolved_key = getpass.getpass(
            "OpenAI API key (input is hidden and is not saved): "
        ).strip()
        if resolved_key:
            return resolved_key

    raise RuntimeError(
        "No OpenAI API key was provided. Set OPENAI_API_KEY or run the command "
        "in an interactive Terminal and enter the key when prompted."
    )


def extract_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def normalize_report_payload(
    payload,
    source_text,
    drafting_allowed=True,
    local_evidence_assessment=None,
):
    validate_report_payload(payload)
    normalized = dict(payload)
    if not drafting_allowed:
        normalized["drafting_decision"] = "needs_more_information"
        normalized["t661_lines"] = {}
        for line_number in ("242", "244", "246"):
            normalized[f"line_{line_number}"] = ""
        if local_evidence_assessment:
            normalized["section_assessments"] = [
                {
                    "line_number": line_number,
                    "status": section["status"],
                    "supported_information": section["supported_information"],
                    "missing_information": section["missing_information"],
                }
                for line_number, section in local_evidence_assessment["sections"].items()
            ]
    elif payload["drafting_decision"] == "draft_ready":
        normalized["t661_lines"] = {
            line_number: build_t661_line(
                line_number,
                payload[f"line_{line_number}"],
            )
            for line_number in ("242", "244", "246")
        }
    else:
        normalized["t661_lines"] = {}

    generated_text = "\n".join(
        payload[f"line_{line_number}"]
        for line_number in ("242", "244", "246")
    )
    new_measurements = find_new_measurements(source_text, generated_text)
    if new_measurements:
        warning = (
            "Verify measurements not found verbatim in the supplied source: "
            + ", ".join(new_measurements)
        )
        normalized["factual_risks"] = [warning, *payload["factual_risks"]]

    return normalized


def validate_report_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("The model report must be a JSON object.")

    missing = [key for key in REPORT_SCHEMA["required"] if key not in payload]
    if missing:
        raise ValueError(f"The model report is missing fields: {', '.join(missing)}")

    if payload["eligibility_signal"] not in ELIGIBILITY_SIGNALS:
        raise ValueError("The model report has an invalid eligibility signal.")

    if payload["structure_mode"] not in STRUCTURE_MODES:
        raise ValueError("The model report has an invalid structure mode.")

    if payload["confidence"] not in CONFIDENCE_LEVELS:
        raise ValueError("The model report has an invalid confidence level.")

    if payload["drafting_decision"] not in DRAFTING_DECISIONS:
        raise ValueError("The model report has an invalid drafting decision.")

    list_fields = [
        "technical_streams",
        "section_assessments",
        "follow_up_questions",
        "factual_risks",
        "review_notes",
    ]
    for field in list_fields:
        if not isinstance(payload[field], list):
            raise ValueError(f"The model report field '{field}' must be a list.")

    line_values = [
        str(payload[f"line_{line_number}"]).strip()
        for line_number in ("242", "244", "246")
    ]
    if payload["drafting_decision"] == "draft_ready" and not all(line_values):
        raise ValueError("A draft-ready model report must populate all T661 lines.")

    if payload["drafting_decision"] == "needs_more_information" and any(line_values):
        raise ValueError(
            "A needs-more-information model report must not contain partial T661 drafts."
        )

    assessed_lines = {
        section.get("line_number")
        for section in payload["section_assessments"]
        if isinstance(section, dict)
    }
    if len(payload["section_assessments"]) != 3 or assessed_lines != {"242", "244", "246"}:
        raise ValueError("The model report must assess Lines 242, 244, and 246 once each.")


def find_new_measurements(source_text, generated_text):
    return find_unsupported_numeric_facts(source_text, generated_text)


def render_llm_report(report, context, model, usage=None, generated_at=None):
    generated_at = generated_at or datetime.now().astimezone()
    local_analysis = context["local_analysis"]
    local_strategy = context["local_strategy"]
    output_status = (
        "Draft for human verification, not an eligibility opinion"
        if report["drafting_decision"] == "draft_ready"
        else "Evidence assessment only; T661 drafting withheld"
    )
    lines = [
        "# AI-Assisted SR&ED Capability Test",
        "",
        f"- Generated: {generated_at.isoformat(timespec='seconds')}",
        f"- Model: `{model}`",
        f"- Local classifier: `{local_analysis['prediction']}`",
        f"- AI eligibility signal: `{report['eligibility_signal']}`",
        f"- AI confidence: `{report['confidence']}`",
        f"- Status: {output_status}",
    ]

    if usage:
        lines.extend([
            f"- Input tokens: {usage.get('input_tokens', 'unavailable')}",
            f"- Output tokens: {usage.get('output_tokens', 'unavailable')}",
            f"- Total tokens: {usage.get('total_tokens', 'unavailable')}",
        ])

    lines.extend([
        "",
        "## Overall Assessment",
        "",
        report["overall_assessment"],
        "",
        "## Structure Decision",
        "",
        f"**AI selection:** `{report['structure_mode']}`",
        "",
        report["structure_rationale"],
        "",
        f"**Local planner selection:** `{local_strategy['selected_structure']['mode']}`",
    ])
    lines.extend(render_bullets(local_strategy["rationale"]))
    lines.extend([
        "",
        "## T661 Evidence Assessment",
        "",
        f"**Drafting decision:** `{report['drafting_decision']}`",
    ])

    for section in report["section_assessments"]:
        lines.extend([
            "",
            f"### Line {section['line_number']}",
            "",
            f"**Status:** `{section['status']}`",
            "",
            "**Information supported by the source:**",
        ])
        lines.extend(render_bullets(section["supported_information"]))
        lines.extend(["", "**Missing information:**"])
        lines.extend(render_bullets(section["missing_information"]))

    if report["drafting_decision"] == "draft_ready":
        lines.extend(["", "## T661 Project Description Draft"])
        for line_number in ("242", "244", "246"):
            line = report["t661_lines"][line_number]
            lines.extend([
                "",
                f"### Line {line_number}",
                "",
                T661_LINE_TITLES[line_number],
                "",
                f"**Word count:** {line['word_count']} / {line['word_limit']}",
                "",
                line["draft"],
            ])
            if line["warnings"]:
                lines.extend(["", "**Warnings:**"])
                lines.extend(f"- {warning}" for warning in line["warnings"])
        if report.get("draft_support"):
            lines.extend(["", "## Draft Evidence Map"])
            for support in report["draft_support"]:
                evidence_ids = ", ".join(support["evidence_ids"]) or "None"
                lines.extend([
                    "",
                    f"### Line {support['line_number']}",
                    "",
                    f"**Evidence IDs:** {evidence_ids}",
                    "",
                    support["coverage_note"],
                ])
        if report.get("claim_support"):
            audit_by_claim = {
                item["claim_id"]: item
                for item in report.get("grounding_audit", {}).get("claims", [])
            }
            lines.extend(["", "## Claim-Level Grounding"])
            for claim in report["claim_support"]:
                audit = audit_by_claim.get(claim["claim_id"], {})
                evidence_ids = ", ".join(claim["evidence_ids"])
                lines.extend([
                    "",
                    f"### {claim['claim_id']} - Line {claim['line_number']}",
                    "",
                    claim["claim_text"],
                    "",
                    f"- Evidence IDs: {evidence_ids}",
                    f"- Independent audit: `{audit.get('verdict', 'not_run')}`",
                    f"- Audit reason: {audit.get('reason', 'Not available.')}",
                ])
    else:
        audit_failed = report.get("draft_audit", {}).get("status") == "failed"
        withholding_message = (
            "**Draft withheld by post-draft audit.** The source evidence passed the "
            "readiness gate, but the generated prose failed a grounding control. Retry "
            "drafting or review the factual risk below; no client fact should be inferred."
            if audit_failed
            else "**Draft not generated.** Resolve the missing information above before drafting."
        )
        lines.extend([
            "",
            "## T661 Drafting Decision",
            "",
            withholding_message,
        ])

    evidence_audit = report.get("evidence_audit")
    evidence_audit_status = report.get("evidence_audit_status", {})
    if evidence_audit or evidence_audit_status.get("status") == "failed":
        lines.extend([
            "",
            "## Independent Evidence Audit",
            "",
            f"**Status:** `{evidence_audit_status.get('status', 'not_run')}`",
        ])
        if evidence_audit_status.get("message"):
            lines.extend(["", evidence_audit_status["message"]])
        if evidence_audit:
            item_audits = evidence_audit.get("item_audits", [])
            sequence_audits = evidence_audit.get("sequence_audits", [])
            issue_audits = evidence_audit.get("issue_audits", [])
            rejected_ids = [
                item["evidence_id"]
                for item in item_audits
                if any(
                    item.get(dimension) != "supported"
                    for dimension in (
                        "normalized_fact",
                        "category",
                        "certainty",
                        "tax_year_scope",
                        "attribution",
                        "stream_assignment",
                    )
                )
            ]
            rejected_sequence_ids = [
                item["sequence_id"]
                for item in sequence_audits
                if any(
                    item.get(dimension) != "supported"
                    for dimension in ("relationship", "chronology")
                )
            ]
            tax_year_audit = evidence_audit.get("claimed_tax_year_audit", {})
            rejected_issue_ids = [
                item["issue_id"]
                for item in issue_audits
                if item.get("verdict") != "supported"
            ]
            lines.extend([
                "",
                evidence_audit.get("overall_assessment", "No assessment supplied."),
                "",
                f"- Evidence items reviewed: {len(item_audits)}",
                "- Semantically rejected items: "
                + (", ".join(rejected_ids) if rejected_ids else "None"),
                "- Claimed tax year: `"
                + tax_year_audit.get("verdict", "not_reviewed")
                + "`",
                f"- Investigation sequences reviewed: {len(sequence_audits)}",
                "- Rejected investigation sequences: "
                + (
                    ", ".join(rejected_sequence_ids)
                    if rejected_sequence_ids
                    else "None"
                ),
                f"- Extracted blocker issues reviewed: {len(issue_audits)}",
                "- Rejected blocker issues: "
                + (", ".join(rejected_issue_ids) if rejected_issue_ids else "None"),
                "- New contradictions: "
                + str(len(evidence_audit.get("discovered_contradictions", []))),
            ])

    graph = report.get("evidence_graph")
    if graph:
        validation = graph["validation"]
        lines.extend([
            "",
            "## Validated Evidence Ledger",
            "",
            f"- Accepted evidence items: {validation['accepted_evidence_items']}",
            f"- Rejected evidence items: {validation['rejected_evidence_items']}",
            "- Accepted investigation sequences: "
            + str(validation.get("accepted_investigation_sequences", 0)),
            "- Rejected investigation sequences: "
            + str(validation.get("rejected_investigation_sequences", 0)),
            "- Accepted extracted blocker issues: "
            + str(validation.get("accepted_extracted_issues", 0)),
            "- Rejected extracted blocker issues: "
            + str(validation.get("rejected_extracted_issues", 0)),
        ])
        for item in graph["evidence_items"]:
            lines.extend([
                "",
                f"### {item['id']} - {item['stream_id']} / {item['category']}",
                "",
                f"- Certainty: `{item['certainty']}`",
                f"- Tax-year scope: `{item['tax_year_scope']}`",
                f"- Attribution: `{item['attribution']}`",
                f"- Source location: {item['source_location'] or 'Not supplied'}",
                f"- Source quote: \"{item['source_quote']}\"",
                f"- Normalized fact: {item['normalized_fact']}",
            ])
        if graph.get("investigation_sequences"):
            lines.extend(["", "## Validated Investigation Sequences"])
            for sequence in graph["investigation_sequences"]:
                lines.extend([
                    "",
                    f"### {sequence['id']} - {sequence['stream_id']} / {sequence['title']}",
                    "",
                    "- Uncertainty: "
                    + ", ".join(sequence["uncertainty_evidence_ids"]),
                    "- Hypothesis: "
                    + ", ".join(sequence["hypothesis_evidence_ids"]),
                    "- Experiment or analysis: "
                    + ", ".join(sequence["experiment_evidence_ids"]),
                    "- Result: " + ", ".join(sequence["result_evidence_ids"]),
                    "- Conclusion: "
                    + ", ".join(sequence["conclusion_evidence_ids"]),
                    "- Advancement: "
                    + ", ".join(sequence["advancement_evidence_ids"]),
                ])

    if report.get("agent_stages"):
        lines.extend(["", "## Agent Stages"])
        lines.extend(
            f"- `{item['stage']}`: `{item['status']}`"
            for item in report["agent_stages"]
        )

    lines.extend(["", "## Technical Streams"])
    for stream in report["technical_streams"]:
        lines.extend([
            "",
            f"### {stream['id']} - {stream['title']}",
            "",
            f"**Uncertainty:** {stream['uncertainty']}",
            "",
            f"**Standard-practice gap:** {stream['standard_practice_gap']}",
            "",
            f"**Systematic investigation:** {stream['systematic_investigation']}",
            "",
            f"**Advancement:** {stream['advancement']}",
            "",
            "**Source support:**",
        ])
        lines.extend(render_bullets(stream["source_support"]))
        lines.extend(["", "**Evidence gaps:**"])
        lines.extend(render_bullets(stream["evidence_gaps"]))

    lines.extend(["", "## Follow-Up Questions"])
    for index, item in enumerate(report["follow_up_questions"], start=1):
        lines.extend([
            "",
            f"### {index}. {item['question']}",
            "",
            item["why_it_matters"],
            "",
            "Records or details to check:",
        ])
        lines.extend(render_bullets(item["examples_to_check"]))

    lines.extend(["", "## Factual Risks"])
    lines.extend(render_bullets(report["factual_risks"]))
    lines.extend(["", "## Human Review Notes"])
    lines.extend(render_bullets(report["review_notes"]))
    lines.append("")

    return "\n".join(lines)


def render_bullets(items):
    if not items:
        return ["- None identified."]
    return [f"- {item}" for item in items]


def save_report(report_text, source_path, output_path=None, base_dir=BASE_DIR):
    if output_path:
        path = Path(output_path)
    else:
        report_dir = Path(base_dir) / OUTPUT_DIR
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        source_name = sanitize_filename(Path(source_path).stem or "project")
        path = report_dir / f"{source_name}_ai_report_{timestamp}.md"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a local AI-assisted SR&ED report capability test."
    )
    parser.add_argument("input_path", help="Path to a UTF-8 project-description file.")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model ID. Defaults to OPENAI_MODEL or %(default)s.",
    )
    parser.add_argument("--output", help="Optional Markdown output path.")
    parser.add_argument("--show", action="store_true", help="Print the generated report.")
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default=os.environ.get("OPENAI_REASONING_EFFORT", "high"),
        help="Reasoning effort for extraction, drafting, and audit (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run local analysis without making an OpenAI API request.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    source_path = Path(args.input_path).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"Input file not found: {source_path}")

    source_text = source_path.read_text(encoding="utf-8").strip()
    if not source_text:
        raise SystemExit("Input file is empty.")

    print("Running local classifier and CRA evidence analysis...")
    context = build_local_context(source_text)
    print(f"Local classification: {context['local_analysis']['prediction']}")
    print(
        "Local structure: "
        f"{context['local_strategy']['selected_structure']['mode']}"
    )

    if args.dry_run:
        print("Dry run complete. No OpenAI API request was made.")
        return

    print(f"Extracting and validating project evidence with {args.model}...")
    try:
        report, usage = request_llm_report(
            context,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report_text = render_llm_report(report, context, args.model, usage=usage)
    report_path = save_report(
        report_text,
        source_path,
        output_path=args.output,
    )
    print(f"Saved report: {report_path}")
    if usage:
        print(
            "Token usage: "
            f"input={usage.get('input_tokens', 'unavailable')}, "
            f"output={usage.get('output_tokens', 'unavailable')}, "
            f"total={usage.get('total_tokens', 'unavailable')}"
        )

    if args.show:
        print("\n" + report_text)


if __name__ == "__main__":
    main()

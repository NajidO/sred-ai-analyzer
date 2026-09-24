import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re

from analysis_engine import BASE_DIR, load_classifier
from evidence_agent import EVIDENCE_CATEGORIES
from llm_report_agent import (
    DEFAULT_MODEL,
    ELIGIBILITY_SIGNALS,
    REASONING_EFFORTS,
    STRUCTURE_MODES,
    build_local_context,
    build_responses_client,
    render_llm_report,
    request_llm_report,
    resolve_api_key,
)


DEFAULT_BENCHMARK = BASE_DIR / "benchmarks" / "live_agent_benchmark.json"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "benchmark_results"
T661_LINES = ("242", "244", "246")
DRAFT_READY = "draft_ready"
NEEDS_MORE_INFORMATION = "needs_more_information"
BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_SUITES = {"smoke", "full"}


def validate_benchmark_data(benchmark_data):
    if not isinstance(benchmark_data, dict):
        raise ValueError("The live benchmark must be a JSON object.")
    if not isinstance(benchmark_data.get("name"), str) or not benchmark_data[
        "name"
    ].strip():
        raise ValueError("The live benchmark requires a non-empty name.")
    if benchmark_data.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"The live benchmark requires schema_version {BENCHMARK_SCHEMA_VERSION}."
        )
    if not isinstance(benchmark_data.get("description"), str) or not benchmark_data[
        "description"
    ].strip():
        raise ValueError("The live benchmark requires a non-empty description.")
    cases = benchmark_data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The live benchmark requires at least one case.")

    seen_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every live benchmark case must be an object.")
        required = {"id", "category", "description", "smoke", "text", "expected"}
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(
                "A live benchmark case is missing fields: " + ", ".join(missing)
            )
        case_id = case["id"]
        if not isinstance(case_id, str) or not re.fullmatch(
            r"[a-z0-9]+(?:_[a-z0-9]+)*",
            case_id,
        ):
            raise ValueError(f"Invalid live benchmark case ID: {case_id!r}.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate live benchmark case ID: {case_id}.")
        seen_ids.add(case_id)
        if not isinstance(case["category"], str) or not case["category"].strip():
            raise ValueError(f"Case {case_id} requires a category.")
        if not isinstance(case["description"], str) or not case["description"].strip():
            raise ValueError(f"Case {case_id} requires a description.")
        if not isinstance(case["smoke"], bool):
            raise ValueError(f"Case {case_id} field smoke must be boolean.")
        text = case["text"]
        if isinstance(text, list):
            if not text or not all(isinstance(item, str) and item.strip() for item in text):
                raise ValueError(f"Case {case_id} has invalid text sections.")
        elif not isinstance(text, str) or not text.strip():
            raise ValueError(f"Case {case_id} requires non-empty text.")

        expected = case["expected"]
        if not isinstance(expected, dict):
            raise ValueError(f"Case {case_id} expected result must be an object.")
        decision = expected.get("drafting_decision")
        if decision not in {DRAFT_READY, NEEDS_MORE_INFORMATION}:
            raise ValueError(
                f"Case {case_id} has invalid expected drafting decision {decision!r}."
            )
        for field in (
            "blocked_lines_include",
            "blocked_lines_exclude",
            "required_evidence_categories",
            "required_claimed_year_evidence_categories",
            "forbidden_claimed_year_evidence_categories",
            "structure_mode_one_of",
            "eligibility_signal_one_of",
        ):
            value = expected.get(field, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError(
                    f"Case {case_id} field {field} must be a non-empty string list."
                )
        included_lines = set(expected.get("blocked_lines_include", []))
        excluded_lines = set(expected.get("blocked_lines_exclude", []))
        invalid_lines = (included_lines | excluded_lines) - set(T661_LINES)
        if invalid_lines:
            raise ValueError(
                f"Case {case_id} references invalid T661 lines: "
                + ", ".join(sorted(invalid_lines))
                + "."
            )
        overlapping_lines = included_lines & excluded_lines
        if overlapping_lines:
            raise ValueError(
                f"Case {case_id} requires and excludes the same blocked lines: "
                + ", ".join(sorted(overlapping_lines))
                + "."
            )
        if decision == DRAFT_READY and set(T661_LINES) != excluded_lines:
            raise ValueError(
                f"Draft-ready case {case_id} must require all T661 lines to be ready."
            )

        expected_categories = {
            *expected.get("required_evidence_categories", []),
            *expected.get("required_claimed_year_evidence_categories", []),
            *expected.get("forbidden_claimed_year_evidence_categories", []),
        }
        invalid_categories = expected_categories - EVIDENCE_CATEGORIES
        if invalid_categories:
            raise ValueError(
                f"Case {case_id} references invalid evidence categories: "
                + ", ".join(sorted(invalid_categories))
                + "."
            )
        conflicting_categories = set(
            expected.get("required_claimed_year_evidence_categories", [])
        ) & set(expected.get("forbidden_claimed_year_evidence_categories", []))
        if conflicting_categories:
            raise ValueError(
                f"Case {case_id} both requires and forbids claimed-year categories: "
                + ", ".join(sorted(conflicting_categories))
                + "."
            )
        allowed_modes = expected.get("structure_mode_one_of", [])
        if not allowed_modes or set(allowed_modes) - STRUCTURE_MODES:
            raise ValueError(f"Case {case_id} requires valid structure modes.")
        allowed_signals = expected.get("eligibility_signal_one_of", [])
        if not allowed_signals or set(allowed_signals) - ELIGIBILITY_SIGNALS:
            raise ValueError(f"Case {case_id} requires valid eligibility signals.")

        expected_year = expected.get("claimed_tax_year")
        if not isinstance(expected_year, str) or (
            expected_year and not re.fullmatch(r"\d{4}", expected_year)
        ):
            raise ValueError(
                f"Case {case_id} claimed tax year must be empty or four digits."
            )
        question_groups = expected.get("required_question_term_groups", [])
        if not isinstance(question_groups, list) or not all(
            isinstance(group, list)
            and group
            and all(isinstance(term, str) and term.strip() for term in group)
            for group in question_groups
        ):
            raise ValueError(
                f"Case {case_id} field required_question_term_groups must contain "
                "non-empty string lists."
            )
        for field in (
            "minimum_streams",
            "maximum_streams",
            "minimum_sequences",
            "maximum_sequences",
            "minimum_contradictions",
            "minimum_material_conflicts",
            "minimum_follow_up_questions",
        ):
            value = expected.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"Case {case_id} field {field} must be non-negative.")
        for minimum_field, maximum_field in (
            ("minimum_streams", "maximum_streams"),
            ("minimum_sequences", "maximum_sequences"),
        ):
            minimum = expected.get(minimum_field)
            maximum = expected.get(maximum_field)
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(
                    f"Case {case_id} field {minimum_field} exceeds {maximum_field}."
                )
    if not any(case["smoke"] for case in cases):
        raise ValueError("The live benchmark requires at least one smoke case.")


def select_benchmark_cases(benchmark_data, selected_case_ids=None, suite="smoke"):
    if suite not in BENCHMARK_SUITES:
        raise ValueError(
            "Live benchmark suite must be one of: " + ", ".join(sorted(BENCHMARK_SUITES))
        )
    selected = set(selected_case_ids or [])
    case_ids = {case["id"] for case in benchmark_data["cases"]}
    unknown = selected - case_ids
    if unknown:
        raise ValueError("Unknown live benchmark case IDs: " + ", ".join(sorted(unknown)))
    if selected:
        return [case for case in benchmark_data["cases"] if case["id"] in selected]
    if suite == "smoke":
        return [case for case in benchmark_data["cases"] if case["smoke"]]
    return list(benchmark_data["cases"])


def case_text(case):
    text = case["text"]
    return "\n\n".join(text) if isinstance(text, list) else text


def evaluate_report(case, report):
    expected = case["expected"]
    failures = []
    decision = report.get("drafting_decision")
    if decision != expected["drafting_decision"]:
        failures.append(
            f"expected drafting decision {expected['drafting_decision']}, "
            f"observed {decision}"
        )

    raw_sections = report.get("section_assessments", [])
    sections = {
        item.get("line_number"): item
        for item in raw_sections
        if isinstance(item, dict)
    }
    if len(raw_sections) != len(T661_LINES) or set(sections) != set(T661_LINES):
        failures.append("report did not contain exactly one assessment for each T661 line")
    blocked_lines = {
        line_number
        for line_number, section in sections.items()
        if section.get("status") != "ready"
    }
    for line_number in expected.get("blocked_lines_include", []):
        if line_number not in blocked_lines:
            failures.append(f"expected Line {line_number} to be blocked")
    for line_number in expected.get("blocked_lines_exclude", []):
        if line_number in blocked_lines:
            failures.append(f"expected Line {line_number} not to be blocked")

    graph = report.get("evidence_graph") or {}
    streams = graph.get("technical_streams", [])
    sequences = graph.get("investigation_sequences", [])
    contradictions = graph.get("contradictions", [])
    attribution_issues = graph.get("attribution_issues", [])
    categories = {
        item.get("category")
        for item in graph.get("evidence_items", [])
        if isinstance(item, dict)
    }
    claimed_year_categories = {
        item.get("category")
        for item in graph.get("evidence_items", [])
        if isinstance(item, dict) and item.get("tax_year_scope") == "claimed_year"
    }
    expected_year = expected.get("claimed_tax_year")
    if graph.get("claimed_tax_year") != expected_year:
        failures.append(
            f"expected claimed tax year {expected_year}, "
            f"observed {graph.get('claimed_tax_year')!r}"
        )
    for category in expected.get("required_evidence_categories", []):
        if category not in categories:
            failures.append(f"missing accepted evidence category {category}")
    for category in expected.get("required_claimed_year_evidence_categories", []):
        if category not in claimed_year_categories:
            failures.append(f"missing claimed-year evidence category {category}")
    for category in expected.get("forbidden_claimed_year_evidence_categories", []):
        if category in claimed_year_categories:
            failures.append(f"accepted forbidden claimed-year evidence category {category}")

    minimum_streams = expected.get("minimum_streams")
    maximum_streams = expected.get("maximum_streams")
    minimum_sequences = expected.get("minimum_sequences")
    maximum_sequences = expected.get("maximum_sequences")
    minimum_contradictions = expected.get("minimum_contradictions")
    minimum_material_conflicts = expected.get("minimum_material_conflicts")
    if minimum_streams is not None and len(streams) < minimum_streams:
        failures.append(
            f"expected at least {minimum_streams} streams, observed {len(streams)}"
        )
    if maximum_streams is not None and len(streams) > maximum_streams:
        failures.append(
            f"expected at most {maximum_streams} streams, observed {len(streams)}"
        )
    if minimum_sequences is not None and len(sequences) < minimum_sequences:
        failures.append(
            f"expected at least {minimum_sequences} sequences, observed {len(sequences)}"
        )
    if maximum_sequences is not None and len(sequences) > maximum_sequences:
        failures.append(
            f"expected at most {maximum_sequences} sequences, observed {len(sequences)}"
        )
    if minimum_contradictions is not None and len(contradictions) < minimum_contradictions:
        failures.append(
            "expected at least "
            f"{minimum_contradictions} contradictions, observed {len(contradictions)}"
        )
    material_conflicts = len(contradictions) + len(attribution_issues)
    if (
        minimum_material_conflicts is not None
        and material_conflicts < minimum_material_conflicts
    ):
        failures.append(
            "expected at least "
            f"{minimum_material_conflicts} material conflicts, observed "
            f"{material_conflicts}"
        )

    allowed_modes = expected.get("structure_mode_one_of", [])
    if allowed_modes and report.get("structure_mode") not in allowed_modes:
        failures.append(
            f"expected structure mode in {allowed_modes}, "
            f"observed {report.get('structure_mode')}"
        )
    allowed_signals = expected.get("eligibility_signal_one_of", [])
    if allowed_signals and report.get("eligibility_signal") not in allowed_signals:
        failures.append(
            f"expected eligibility signal in {allowed_signals}, "
            f"observed {report.get('eligibility_signal')}"
        )

    questions = [
        item
        for item in report.get("follow_up_questions", [])
        if isinstance(item, dict) and str(item.get("question", "")).strip()
    ]
    minimum_questions = expected.get("minimum_follow_up_questions")
    if minimum_questions is not None and len(questions) < minimum_questions:
        failures.append(
            f"expected at least {minimum_questions} follow-up questions, "
            f"observed {len(questions)}"
        )
    question_text = " ".join(
        str(item.get("question", ""))
        for item in questions
        if isinstance(item, dict)
    ).casefold()
    for term_group in expected.get("required_question_term_groups", []):
        if not any(term.casefold() in question_text for term in term_group):
            failures.append(
                "follow-up questions omitted expected concept (one of: "
                + ", ".join(repr(term) for term in term_group)
                + ")"
            )

    raw_t661_lines = report.get("t661_lines", {})
    t661_lines = raw_t661_lines
    if not isinstance(t661_lines, dict):
        failures.append("report returned an invalid T661 line collection")
        t661_lines = {}
    line_values = [str(report.get(f"line_{line}", "")).strip() for line in T661_LINES]
    draft_artifacts_present = bool(
        raw_t661_lines or any(line_values) or report.get("claim_support")
    )
    if decision == DRAFT_READY:
        line_objects = [t661_lines.get(line) for line in T661_LINES]
        mapped_line_values = [
            str(item.get("draft", "")).strip() if isinstance(item, dict) else ""
            for item in line_objects
        ]
        if (
            set(t661_lines) != set(T661_LINES)
            or not all(isinstance(item, dict) for item in line_objects)
            or not all(line_values)
            or not all(mapped_line_values)
        ):
            failures.append("draft-ready report did not contain all three T661 lines")
        elif mapped_line_values != line_values:
            failures.append("draft-ready report returned inconsistent T661 line fields")
        if not report.get("claim_support"):
            failures.append("draft-ready report returned no claim-level support")
        else:
            supported_lines = {
                item.get("line_number")
                for item in report["claim_support"]
                if isinstance(item, dict)
            }
            missing_support = set(T661_LINES) - supported_lines
            if missing_support:
                failures.append(
                    "draft-ready report omitted claim support for Lines "
                    + ", ".join(sorted(missing_support))
                )
        stages = {
            item.get("stage"): item.get("status")
            for item in report.get("agent_stages", [])
            if isinstance(item, dict)
        }
        for stage in (
            "independent_evidence_audit",
            "readiness_gate",
            "grounded_drafting",
            "post_draft_local_audit",
            "independent_grounding_audit",
        ):
            if stages.get(stage) not in {"passed", "complete"}:
                failures.append(f"draft-ready report stage {stage} did not pass")
    elif draft_artifacts_present:
        failures.append("blocked report retained partial T661 prose or claim support")

    return {
        "passed": not failures,
        "failures": failures,
        "observed": {
            "drafting_decision": decision,
            "blocked_lines": sorted(blocked_lines),
            "eligibility_signal": report.get("eligibility_signal"),
            "structure_mode": report.get("structure_mode"),
            "streams": len(streams),
            "sequences": len(sequences),
            "contradictions": len(contradictions),
            "material_conflicts": material_conflicts,
            "claimed_tax_year": graph.get("claimed_tax_year"),
            "accepted_evidence_items": graph.get("validation", {}).get(
                "accepted_evidence_items",
                0,
            ),
            "follow_up_questions": len(questions),
            "draft_artifacts_present": draft_artifacts_present,
        },
    }


def summarize_safety_metrics(results):
    expected_ready = [
        item for item in results if item["expected_decision"] == DRAFT_READY
    ]
    expected_blocked = [
        item
        for item in results
        if item["expected_decision"] == NEEDS_MORE_INFORMATION
    ]
    false_ready = [
        item
        for item in expected_blocked
        if item.get("observed", {}).get("drafting_decision") == DRAFT_READY
        or item.get("observed", {}).get("draft_artifacts_present")
    ]
    false_block = [
        item
        for item in expected_ready
        if item.get("observed", {}).get("drafting_decision") != DRAFT_READY
    ]
    return {
        "expected_ready_cases": len(expected_ready),
        "expected_blocked_cases": len(expected_blocked),
        "false_ready_cases": len(false_ready),
        "false_ready_case_ids": [item["id"] for item in false_ready],
        "false_block_cases": len(false_block),
        "false_block_case_ids": [item["id"] for item in false_block],
        "failed_safety_controls": sum(not item["passed"] for item in expected_blocked),
        "failed_quality_controls": sum(not item["passed"] for item in expected_ready),
    }


def combine_usage(total, usage):
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None:
            total[key] = total.get(key, 0) + value


def render_summary(result):
    lines = [
        result["name"],
        "=" * 88,
        f"Model: {result['model']}",
        f"Reasoning effort: {result['reasoning_effort']}",
        f"Suite: {result['suite']}",
        f"Passed: {result['passed']}/{result['total']} ({result['score']:.0%})",
        (
            "Safety metrics: "
            f"false_ready={result['safety_metrics']['false_ready_cases']}, "
            f"false_block={result['safety_metrics']['false_block_cases']}, "
            "failed_blocked_case_controls="
            f"{result['safety_metrics']['failed_safety_controls']}, "
            "failed_ready_case_controls="
            f"{result['safety_metrics']['failed_quality_controls']}"
        ),
        (
            "Token usage: "
            f"input={result['usage'].get('input_tokens', 'unavailable')}, "
            f"output={result['usage'].get('output_tokens', 'unavailable')}, "
            f"total={result['usage'].get('total_tokens', 'unavailable')}"
        ),
    ]
    for item in result["results"]:
        observed = item.get("observed", {})
        lines.extend([
            "",
            f"{'PASS' if item['passed'] else 'FAIL'}: {item['id']} [{item['category']}]",
            item["description"],
            (
                "Observed: "
                f"decision={observed.get('drafting_decision', 'error')}, "
                f"blocked={observed.get('blocked_lines', [])}, "
                f"streams={observed.get('streams', 0)}, "
                f"sequences={observed.get('sequences', 0)}, "
                f"structure={observed.get('structure_mode', 'unavailable')}"
            ),
        ])
        lines.extend(f"Failure: {failure}" for failure in item["failures"])
        if item.get("error"):
            lines.append("Error: " + item["error"])
        if item.get("report_path"):
            lines.append("Report: " + item["report_path"])
    return "\n".join(lines)


def run_live_benchmark(
    benchmark_data,
    model,
    reasoning_effort,
    client,
    classifier,
    output_dir,
    selected_case_ids=None,
    suite="smoke",
):
    validate_benchmark_data(benchmark_data)
    cases = select_benchmark_cases(
        benchmark_data,
        selected_case_ids=selected_case_ids,
        suite=suite,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    usage_total = {}
    results = []
    for case in cases:
        source_text = case_text(case)
        try:
            context = build_local_context(source_text, classifier=classifier)
            report, usage = request_llm_report(
                context,
                model=model,
                reasoning_effort=reasoning_effort,
                client=client,
            )
            evaluation = evaluate_report(case, report)
            combine_usage(usage_total, usage)
            report_text = render_llm_report(report, context, model, usage=usage)
            report_path = output_dir / f"{case['id']}.md"
            json_path = output_dir / f"{case['id']}.json"
            report_path.write_text(report_text, encoding="utf-8")
            json_path.write_text(
                json.dumps(
                    {"case": case, "report": report, "usage": usage},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            results.append({
                "id": case["id"],
                "category": case["category"],
                "description": case["description"],
                "expected_decision": case["expected"]["drafting_decision"],
                **evaluation,
                "usage": usage,
                "report_path": str(report_path.resolve()),
                "json_path": str(json_path.resolve()),
            })
        except Exception as exc:
            report_path = output_dir / f"{case['id']}.md"
            json_path = output_dir / f"{case['id']}.json"
            error_message = str(exc)
            report_path.write_text(
                "\n".join([
                    "# Live Agent Benchmark Execution Failure",
                    "",
                    f"- Case: `{case['id']}`",
                    f"- Error type: `{type(exc).__name__}`",
                    "",
                    error_message,
                    "",
                ]),
                encoding="utf-8",
            )
            json_path.write_text(
                json.dumps(
                    {
                        "case": case,
                        "error": {
                            "type": type(exc).__name__,
                            "message": error_message,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            results.append({
                "id": case["id"],
                "category": case["category"],
                "description": case["description"],
                "expected_decision": case["expected"]["drafting_decision"],
                "passed": False,
                "failures": ["agent execution failed"],
                "observed": {},
                "error": error_message,
                "usage": {},
                "report_path": str(report_path.resolve()),
                "json_path": str(json_path.resolve()),
            })

    passed = sum(item["passed"] for item in results)
    result = {
        "name": benchmark_data["name"],
        "model": model,
        "reasoning_effort": reasoning_effort,
        "suite": "selected" if selected_case_ids else suite,
        "passed": passed,
        "total": len(results),
        "score": passed / len(results) if results else 0,
        "usage": usage_total,
        "results": results,
    }
    result["safety_metrics"] = summarize_safety_metrics(results)
    summary_text = render_summary(result)
    (output_dir / "summary.md").write_text(summary_text + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run reproducible live-model benchmarks against the full SR&ED agent."
    )
    parser.add_argument(
        "benchmark_path",
        nargs="?",
        default=str(DEFAULT_BENCHMARK),
        help="Live benchmark JSON path.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model ID. Defaults to OPENAI_MODEL or %(default)s.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default=os.environ.get("OPENAI_REASONING_EFFORT", "high"),
        help="Reasoning effort for every model stage.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run one case ID; repeat to select several cases.",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(BENCHMARK_SUITES),
        default="smoke",
        help="Run smoke cases or the full extended set (default: %(default)s).",
    )
    parser.add_argument("--output-dir", help="Directory for reports and result JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fixtures and run local preparation without API requests.",
    )
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Return zero even when one or more live cases fail.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    validate_benchmark_data(benchmark_data)

    try:
        cases = select_benchmark_cases(
            benchmark_data,
            selected_case_ids=args.case_ids,
            suite=args.suite,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    classifier = load_classifier()
    if args.dry_run:
        for case in cases:
            build_local_context(case_text(case), classifier=classifier)
        print(
            f"Validated {len(cases)} live benchmark case(s); "
            "no OpenAI API request was made."
        )
        return

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"live_agent_{timestamp}"
    )
    try:
        api_key = resolve_api_key()
        client = build_responses_client(api_key)
        result = run_live_benchmark(
            benchmark_data,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            client=client,
            classifier=classifier,
            output_dir=output_dir,
            selected_case_ids=args.case_ids,
            suite=args.suite,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(render_summary(result))
    print(f"\nSaved benchmark artifacts: {output_dir.resolve()}")
    if result["passed"] != result["total"] and not args.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

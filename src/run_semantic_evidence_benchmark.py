import argparse
from copy import deepcopy
import json
from pathlib import Path

from evidence_agent import (
    EVIDENCE_AUDIT_DIMENSIONS,
    SEQUENCE_AUDIT_DIMENSIONS,
    apply_evidence_audit,
    assess_evidence_graph,
    build_evidence_structure_plan,
    normalize_evidence_graph,
    validate_evidence_audit_payload,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = (
    BASE_DIR / "benchmarks" / "semantic_evidence_gate_benchmark.json"
)


def apply_mutations(source_text, payload, mutations):
    payload = deepcopy(payload)
    for mutation in mutations:
        operation = mutation["op"]
        if operation == "append_source":
            source_text += mutation["value"]
            continue
        if operation != "set":
            raise ValueError(f"Unknown benchmark mutation operation: {operation}")
        target = payload
        path = mutation["path"]
        if not path:
            raise ValueError("A set mutation requires a non-empty path.")
        for segment in path[:-1]:
            target = target[segment]
        target[path[-1]] = mutation["value"]
    return source_text, payload


def build_supporting_audit(graph, audit_settings):
    tax_year_verdict = audit_settings.get(
        "tax_year_verdict",
        "supported" if graph["claimed_tax_year"] else "unsupported",
    )
    sequence_overrides = audit_settings.get("sequence_overrides", {})
    issue_overrides = audit_settings.get("issue_overrides", {})
    sequence_audits = []
    for sequence in graph["investigation_sequences"]:
        overrides = sequence_overrides.get(sequence["id"], {})
        sequence_audits.append({
            "sequence_id": sequence["id"],
            **{
                dimension: overrides.get(dimension, "supported")
                for dimension in SEQUENCE_AUDIT_DIMENSIONS
            },
            "reason": overrides.get(
                "reason",
                "The source supports the proposed relationship and chronology.",
            ),
        })
    return {
        "overall_assessment": "Benchmark semantic audit.",
        "claimed_tax_year_audit": {
            "verdict": tax_year_verdict,
            "reason": audit_settings.get(
                "tax_year_reason",
                "Benchmark review of the claimed tax year.",
            ),
        },
        "item_audits": [
            {
                "evidence_id": item["id"],
                **{
                    dimension: "supported"
                    for dimension in EVIDENCE_AUDIT_DIMENSIONS
                },
                "reason": "The source supports every audited evidence dimension.",
            }
            for item in graph["evidence_items"]
        ],
        "sequence_audits": sequence_audits,
        "issue_audits": [
            {
                "issue_id": issue["id"],
                "verdict": issue_overrides.get(issue["id"], {}).get(
                    "verdict",
                    "supported",
                ),
                "reason": issue_overrides.get(issue["id"], {}).get(
                    "reason",
                    "The source supports the extracted blocker interpretation.",
                ),
            }
            for issue in [*graph["routine_work"], *graph["attribution_issues"]]
        ],
        "discovered_contradictions": [],
    }


def evaluate_case(case, base_source, base_payload):
    source_text, payload = apply_mutations(
        base_source,
        base_payload,
        case.get("mutations", []),
    )
    graph = normalize_evidence_graph(payload, source_text)
    if "audit" in case:
        audit = build_supporting_audit(graph, case["audit"])
        validate_evidence_audit_payload(audit, graph)
        graph = apply_evidence_audit(graph, audit)
    readiness = assess_evidence_graph(graph)
    expected = case["expected"]
    failures = []

    if readiness["can_draft"] != expected["can_draft"]:
        failures.append(
            f"expected can_draft={expected['can_draft']}, "
            f"observed {readiness['can_draft']}"
        )
    blocked_lines = set(readiness["blocked_lines"])
    expected_blocked = set(expected.get("blocked_lines", []))
    if blocked_lines != expected_blocked:
        failures.append(
            f"expected blocked lines {sorted(expected_blocked)}, "
            f"observed {sorted(blocked_lines)}"
        )
    for line_number in expected.get("ready_lines_include", []):
        if readiness["sections"][line_number]["status"] != "ready":
            failures.append(f"expected Line {line_number} to remain ready")

    year_present = bool(graph["claimed_tax_year"])
    if year_present != expected.get("claimed_tax_year_present", True):
        failures.append(
            "claimed tax year presence did not match the expected grounded state"
        )
    accepted_sequences = len(graph["investigation_sequences"])
    if accepted_sequences != expected.get("accepted_sequences", accepted_sequences):
        failures.append(
            f"expected {expected['accepted_sequences']} accepted sequences, "
            f"observed {accepted_sequences}"
        )
    rejected_sequences = graph["validation"].get(
        "rejected_investigation_sequences",
        0,
    )
    minimum_rejected = expected.get("minimum_rejected_sequences", 0)
    if rejected_sequences < minimum_rejected:
        failures.append(
            f"expected at least {minimum_rejected} rejected sequences, "
            f"observed {rejected_sequences}"
        )

    question_categories = {
        item["category"] for item in readiness["follow_up_questions"]
    }
    for category in expected.get("question_categories_include", []):
        if category not in question_categories:
            failures.append(f"missing follow-up question category {category}")

    return {
        "id": case["id"],
        "kind": "evidence_gate",
        "description": case["description"],
        "passed": not failures,
        "failures": failures,
        "observed": {
            "can_draft": readiness["can_draft"],
            "blocked_lines": readiness["blocked_lines"],
            "claimed_tax_year": graph["claimed_tax_year"],
            "accepted_sequences": accepted_sequences,
            "rejected_sequences": rejected_sequences,
            "question_categories": sorted(question_categories),
        },
    }


def evaluate_structure_case(case):
    graph = {
        "technical_streams": case["technical_streams"],
        "investigation_sequences": case["investigation_sequences"],
        "evidence_items": [
            {
                "id": f"GLOBAL_{index}",
                "stream_id": "GLOBAL",
                "category": category,
            }
            for index, category in enumerate(
                case.get("global_context_categories", []),
                start=1,
            )
        ],
    }
    plan = build_evidence_structure_plan(graph)
    expected = case["expected"]
    failures = []
    if plan["mode"] != expected["mode"]:
        failures.append(
            f"expected mode {expected['mode']}, observed {plan['mode']}"
        )
    expected_labels = expected["required_labels_by_line"]
    if plan["required_labels_by_line"] != expected_labels:
        failures.append(
            "required labels differed: expected "
            f"{expected_labels}, observed {plan['required_labels_by_line']}"
        )
    expected_order = expected.get(
        "stream_order",
        [stream["id"] for stream in case["technical_streams"]],
    )
    observed_order = [item["stream_id"] for item in plan["stream_order"]]
    if observed_order != expected_order:
        failures.append(
            f"expected stream order {expected_order}, observed {observed_order}"
        )
    return {
        "id": case["id"],
        "kind": "structure_plan",
        "description": case["description"],
        "passed": not failures,
        "failures": failures,
        "observed": {
            "mode": plan["mode"],
            "required_labels_by_line": plan["required_labels_by_line"],
            "stream_order": observed_order,
        },
    }


def run_benchmark(benchmark_data):
    fixture = benchmark_data["fixture"]
    evidence_results = [
        evaluate_case(case, fixture["source_text"], fixture["evidence_graph"])
        for case in benchmark_data["cases"]
    ]
    structure_results = [
        evaluate_structure_case(case)
        for case in benchmark_data.get("structure_cases", [])
    ]
    results = [*evidence_results, *structure_results]
    passed = sum(result["passed"] for result in results)
    return {
        "name": benchmark_data["name"],
        "passed": passed,
        "total": len(results),
        "score": passed / len(results) if results else 0,
        "results": results,
    }


def render_benchmark(result):
    lines = [
        result["name"],
        "=" * 88,
        f"Passed: {result['passed']}/{result['total']} ({result['score']:.0%})",
    ]
    for case in result["results"]:
        observed = case["observed"]
        lines.extend([
            "",
            f"{'PASS' if case['passed'] else 'FAIL'}: {case['id']}",
            case["description"],
        ])
        if case["kind"] == "structure_plan":
            lines.append(
                "Observed: "
                f"mode={observed['mode']}, "
                f"labels={observed['required_labels_by_line']}, "
                f"order={observed['stream_order']}"
            )
        else:
            lines.append(
                "Observed: "
                f"draft={observed['can_draft']}, "
                f"blocked={observed['blocked_lines']}, "
                f"year={observed['claimed_tax_year'] or 'missing'}, "
                f"sequences={observed['accepted_sequences']}, "
                f"rejected_sequences={observed['rejected_sequences']}"
            )
        lines.extend(f"Failure: {failure}" for failure in case["failures"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial benchmarks against the semantic evidence gate."
    )
    parser.add_argument(
        "benchmark_path",
        nargs="?",
        default=str(DEFAULT_BENCHMARK),
        help="Benchmark JSON path.",
    )
    args = parser.parse_args()
    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    result = run_benchmark(benchmark_data)
    print(render_benchmark(result))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

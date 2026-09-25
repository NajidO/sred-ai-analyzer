import argparse
import json
from pathlib import Path

from analysis_engine import load_classifier
from llm_report_agent import build_local_context, find_new_measurements


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = BASE_DIR / "benchmarks" / "t661_capability_benchmark.json"


def evaluate_case(case, classifier):
    case_text = case["text"]
    if isinstance(case_text, list):
        case_text = "\n\n".join(case_text)

    context = build_local_context(case_text, classifier=classifier)
    assessment = context["t661_evidence_assessment"]
    strategy = context["local_strategy"]
    drafts = context["local_t661_baseline"]
    expected = case["expected"]
    failures = []

    if assessment["can_draft"] != expected["can_draft"]:
        failures.append(
            "draft decision: expected "
            f"{expected['can_draft']}, observed {assessment['can_draft']}"
        )

    blocked_lines = set(assessment["blocked_lines"])
    for line_number in expected.get("blocked_lines_include", []):
        if line_number not in blocked_lines:
            failures.append(f"expected Line {line_number} to be blocked")

    for line_number in expected.get("ready_lines_include", []):
        if assessment["sections"][line_number]["status"] != "ready":
            failures.append(f"expected Line {line_number} to be ready")

    streams = strategy["streams"]
    if len(streams) < expected.get("minimum_streams", 1):
        failures.append(
            f"stream count: expected at least {expected['minimum_streams']}, "
            f"observed {len(streams)}"
        )

    maximum_streams = expected.get("maximum_streams")
    if maximum_streams is not None and len(streams) > maximum_streams:
        failures.append(
            f"stream count: expected at most {maximum_streams}, observed {len(streams)}"
        )

    normalized_titles = " ".join(stream["title"].lower() for stream in streams)
    for excluded_title in expected.get("stream_title_excludes", []):
        if excluded_title.lower() in normalized_titles:
            failures.append(
                f"unexpected stream title content: {excluded_title}"
            )

    generic_stream = any(
        stream["title"] == "Primary technological uncertainty"
        for stream in streams
    )
    if not expected.get("allow_generic_stream", True) and generic_stream:
        failures.append("used a generic fallback stream for a supported technical domain")

    expected_predictions = expected.get("prediction_one_of")
    prediction = context["local_analysis"]["prediction"]
    if expected_predictions and prediction not in expected_predictions:
        failures.append(
            f"classification: expected one of {expected_predictions}, observed {prediction}"
        )

    if assessment["can_draft"]:
        if set(drafts) != {"242", "244", "246"}:
            failures.append("draft-ready case did not produce all three T661 lines")
        else:
            generated_text = "\n".join(
                drafts[line_number]["draft"]
                for line_number in ("242", "244", "246")
            )
            unsupported = find_new_measurements(case_text, generated_text)
            if unsupported:
                failures.append(
                    "draft introduced unsupported measurements: "
                    + ", ".join(unsupported)
                )
    elif drafts:
        failures.append("blocked case produced partial T661 draft content")

    return {
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "passed": not failures,
        "failures": failures,
        "observed": {
            "prediction": prediction,
            "cra_alignment": context["local_analysis"]["cra_alignment"],
            "can_draft": assessment["can_draft"],
            "blocked_lines": assessment["blocked_lines"],
            "structure": strategy["selected_structure"]["mode"],
            "streams": [stream["title"] for stream in streams],
            "missing_information": {
                line_number: section["missing_information"]
                for line_number, section in assessment["sections"].items()
            },
        },
    }


def run_benchmark(benchmark_data, classifier):
    results = [
        evaluate_case(case, classifier)
        for case in benchmark_data["cases"]
    ]
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
        status = "PASS" if case["passed"] else "FAIL"
        observed = case["observed"]
        lines.extend([
            "",
            "-" * 88,
            f"{status}: {case['id']} [{case['category']}]",
            case["description"],
            (
                "Observed: "
                f"classification={observed['prediction']}, "
                f"alignment={observed['cra_alignment']}, "
                f"draft={observed['can_draft']}, "
                f"blocked={observed['blocked_lines']}, "
                f"structure={observed['structure']}"
            ),
            "Streams: " + "; ".join(observed["streams"]),
        ])

        for failure in case["failures"]:
            lines.append(f"Failure: {failure}")

        if case["failures"]:
            for line_number, missing_items in observed["missing_information"].items():
                for item in missing_items:
                    lines.append(f"Observed Line {line_number} gap: {item}")

    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run diverse evidence and drafting benchmarks against the local analyzer."
    )
    parser.add_argument(
        "benchmark_path",
        nargs="?",
        default=str(DEFAULT_BENCHMARK),
        help="Benchmark JSON path.",
    )
    parser.add_argument("--output", help="Optional text output path.")
    parser.add_argument("--json-output", help="Optional machine-readable result path.")
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Return zero even when benchmark cases fail.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    result = run_benchmark(benchmark_data, load_classifier())
    rendered = render_benchmark(result)
    print(rendered)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved text result: {output_path.resolve()}")

    if args.json_output:
        json_output_path = Path(args.json_output).expanduser()
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved JSON result: {json_output_path.resolve()}")

    if result["passed"] != result["total"] and not args.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

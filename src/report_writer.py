from datetime import datetime


def save_report(
    text,
    prediction,
    probabilities,
    labels,
    signals,
    recommendation,
    evidence_map,
    cra_check,
    questions,
    cra_reference_questions,
    base_dir
):
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"sred_analysis_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("SR&ED Technical Uncertainty Analysis Report\n")
        file.write("=" * 60 + "\n\n")

        file.write("Project Description:\n")
        file.write(text + "\n\n")

        file.write("Top Prediction:\n")
        file.write(prediction + "\n\n")

        file.write("Category Probabilities:\n")
        for label, probability in zip(labels, probabilities):
            file.write(f"{label}: {probability:.2f}\n")

        file.write("\nDetected Signals:\n")
        file.write("Uncertainty signals:\n")
        if signals["uncertainty_signals"]:
            for signal in signals["uncertainty_signals"]:
                file.write(f"- {signal}\n")
        else:
            file.write("- None detected\n")

        file.write("\nRoutine signals:\n")
        if signals["routine_signals"]:
            for signal in signals["routine_signals"]:
                file.write(f"- {signal}\n")
        else:
            file.write("- None detected\n")

        file.write("\nRecommendation:\n")
        file.write(recommendation + "\n")

        file.write("\nSR&ED Evidence Map:\n")

        file.write("\nPossible uncertainty:\n")
        file.write(evidence_map["possible_uncertainty"] + "\n")

        file.write("\nPossible experiments:\n")
        for experiment in evidence_map["possible_experiments"]:
            file.write(f"- {experiment}\n")

        file.write("\nPossible results:\n")
        file.write(evidence_map["possible_results"] + "\n")

        file.write("\nMissing evidence to request:\n")
        for item in evidence_map["missing_evidence"]:
            file.write(f"- {item}\n")

        file.write("\nCRA Guideline Alignment Check:\n")
        file.write(f"Overall alignment: {cra_check['overall_alignment']}\n")

        file.write("\nChecklist:\n")
        for check_name, result in cra_check["checks"].items():
            file.write(f"\n{check_name}:\n")
            file.write(f"Status: {result['status']}\n")
            file.write(f"Comment: {result['comment']}\n")

        file.write("\nSuggested follow-up questions:\n")
        for index, question in enumerate(questions, start=1):
            file.write(f"{index}. {question}\n")

        file.write("\nCRA-grounded engagement questions:\n")
        for index, question in enumerate(cra_reference_questions, start=1):
            file.write(f"{index}. {question}\n")

    return report_path
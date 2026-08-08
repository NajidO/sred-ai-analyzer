def map_to_sred_framework(text, signals):
    text_lower = text.lower()

    possible_uncertainty = identify_uncertainty(text_lower, signals)
    possible_experiments = identify_experiments(text_lower, signals)
    possible_results = identify_results(text_lower)
    missing_evidence = suggest_missing_evidence()

    return {
        "possible_uncertainty": possible_uncertainty,
        "possible_experiments": possible_experiments,
        "possible_results": possible_results,
        "missing_evidence": missing_evidence,
    }


def identify_uncertainty(text_lower, signals):
    uncertainty_signals = signals["uncertainty_signals"]

    if "ocr" in text_lower:
        return (
            "Whether the team could reliably extract text from low-quality or variable product labels "
            "where standard OCR tools may have been insufficient."
        )

    if "synchronization" in text_lower or "sync" in text_lower:
        return (
            "Whether the system could maintain data consistency and performance under the required transaction volume."
        )

    if "latency" in text_lower:
        return (
            "Whether the system could meet required latency targets under the relevant operating conditions."
        )

    if "machine learning" in text_lower or "model" in text_lower:
        return (
            "Whether a model or algorithmic approach could meet the required accuracy, reliability, or performance targets."
        )

    if uncertainty_signals:
        return (
            "The description suggests a possible technological uncertainty, but the exact uncertainty needs to be stated more clearly."
        )

    return (
        "No clear technological uncertainty is stated yet."
    )


def identify_experiments(text_lower, signals):
    experiments = []

    if "tested multiple" in text_lower or "tested several" in text_lower:
        experiments.append("Testing multiple technical approaches")

    if "preprocessing" in text_lower:
        experiments.append("Testing preprocessing methods")

    if "model architectures" in text_lower:
        experiments.append("Testing different model architectures")

    if "benchmark" in text_lower:
        experiments.append("Benchmarking or performance measurement")

    if "prototype" in text_lower:
        experiments.append("Prototype development and testing")

    if "configuration" in text_lower or "configurations" in text_lower:
        experiments.append("Testing different technical configurations")

    if not experiments:
        experiments.append("No clear experiments are stated yet. Ask for specific tests, iterations, failures, and results.")

    return experiments


def identify_results(text_lower):
    if "failed" in text_lower:
        return (
            "The description mentions failure, but the specific results of each attempted approach should be documented."
        )

    if "improved" in text_lower or "reduced" in text_lower:
        return (
            "The description mentions improvement, but the before-and-after measurements should be documented."
        )

    return (
        "No clear experimental results are stated yet."
    )


def suggest_missing_evidence():
    return [
        "Jira tickets or sprint notes",
        "Git commits or pull requests",
        "Test logs",
        "Benchmark results",
        "Design documents",
        "Architecture diagrams",
        "Records of failed approaches",
        "Before-and-after performance or accuracy measurements",
    ]
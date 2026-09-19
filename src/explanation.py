def generate_label_explanation(prediction, probabilities, labels, signals, cra_check):
    probability_map = dict(zip(labels, probabilities))
    top_confidence = probability_map.get(prediction, 0)

    reasons = []
    cautions = []
    next_steps = []

    uncertainty_signals = signals["uncertainty_signals"]
    routine_signals = signals["routine_signals"]
    alignment = cra_check["overall_alignment"]
    checks = cra_check["checks"]

    if prediction == "strong_sred":
        reasons.append(
            "The description contains indicators of technological uncertainty and experimental work."
        )
        reasons.append(
            "The model is seeing language that resembles failed standard approaches, testing, iteration, or technical learning."
        )
        next_steps.extend([
            "Confirm the specific uncertainty that could not be resolved using standard practice.",
            "Collect evidence of hypotheses, experiments, failures, benchmark results, and technical conclusions.",
        ])

    elif prediction == "borderline":
        reasons.append(
            "The description contains some technical signals, but the SR&ED story is incomplete."
        )
        reasons.append(
            "It may describe optimization, troubleshooting, or configuration unless the unknown technical limitation is clarified."
        )
        cautions.append(
            "Do not treat this as a strong SR&ED candidate until failed standard methods and experimental learning are documented."
        )
        next_steps.extend([
            "Ask whether known methods were insufficient and why.",
            "Ask what experiments were performed and what was learned from each attempt.",
        ])

    elif prediction == "needs_more_info":
        reasons.append(
            "The description is too general to support a reliable SR&ED assessment."
        )
        reasons.append(
            "It names a technology or improvement but does not yet explain the uncertainty, tests, results, or learning."
        )
        cautions.append(
            "A vague innovation claim is not enough; the technical problem and investigation need to be described."
        )
        next_steps.extend([
            "Ask what was technically unknown at the start of the work.",
            "Ask which standard approaches were considered, tested, or ruled out.",
        ])

    elif prediction == "routine":
        reasons.append(
            "The description resembles standard implementation, configuration, integration, reporting, or use of known tools."
        )
        cautions.append(
            "If there were unexpected technical limits or experiments, they are not clearly described yet."
        )
        next_steps.extend([
            "Confirm whether the work was completed using standard practice.",
            "Ask whether any failed attempts or systematic technical tests occurred.",
        ])

    if uncertainty_signals:
        reasons.append(
            "Detected uncertainty terms: " + ", ".join(uncertainty_signals) + "."
        )

    if routine_signals:
        cautions.append(
            "Detected routine implementation terms: " + ", ".join(routine_signals) + "."
        )

    missing_checks = [
        check_name
        for check_name, result in checks.items()
        if result["status"] == "missing"
    ]
    partial_checks = [
        check_name
        for check_name, result in checks.items()
        if result["status"] == "partial"
    ]

    if alignment == "strong_alignment":
        reasons.append("The CRA guideline checklist shows strong alignment signals.")
    elif alignment == "partial_alignment":
        cautions.append("The CRA guideline checklist shows partial alignment only.")
    else:
        cautions.append("The CRA guideline checklist shows weak alignment.")

    if missing_checks:
        next_steps.append(
            "Clarify missing CRA checklist areas: " + ", ".join(missing_checks) + "."
        )

    if partial_checks:
        next_steps.append(
            "Strengthen partially supported areas: " + ", ".join(partial_checks) + "."
        )

    return {
        "confidence": top_confidence,
        "reasons": reasons,
        "cautions": cautions,
        "next_steps": next_steps,
    }

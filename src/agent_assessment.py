def build_agent_assessment(
    prediction,
    probabilities,
    labels,
    signals,
    cra_check,
    evidence_map,
    questions,
    cra_reference_questions,
    explanation,
):
    probability_map = dict(zip(labels, probabilities))
    confidence = probability_map.get(prediction, 0)
    checks = cra_check["checks"]
    missing_checks = _checks_with_status(checks, "missing")
    partial_checks = _checks_with_status(checks, "partial")
    uncertainty_signals = signals["uncertainty_signals"]
    routine_signals = signals["routine_signals"]

    case_stage = _select_case_stage(
        prediction,
        confidence,
        cra_check["overall_alignment"],
        missing_checks,
        partial_checks,
        uncertainty_signals,
        routine_signals,
    )
    priority = _select_priority(prediction, confidence, case_stage)
    decision = _select_decision(case_stage, prediction)
    blockers = _build_blockers(missing_checks, partial_checks, routine_signals)
    evidence_requests = _prioritize_evidence_requests(evidence_map, checks, prediction)
    interview_focus = _prioritize_questions(questions, cra_reference_questions, missing_checks)
    action_plan = _build_action_plan(case_stage, blockers, evidence_requests, interview_focus)

    return {
        "case_stage": case_stage,
        "priority": priority,
        "decision": decision,
        "confidence": confidence,
        "blockers": blockers,
        "evidence_requests": evidence_requests,
        "interview_focus": interview_focus,
        "action_plan": action_plan,
        "handoff_summary": _build_handoff_summary(
            prediction,
            confidence,
            case_stage,
            cra_check["overall_alignment"],
            blockers,
            explanation,
        ),
    }


def _checks_with_status(checks, status):
    return [
        check_name
        for check_name, result in checks.items()
        if result["status"] == status
    ]


def _select_case_stage(
    prediction,
    confidence,
    alignment,
    missing_checks,
    partial_checks,
    uncertainty_signals,
    routine_signals,
):
    if prediction == "strong_sred" and alignment == "strong_alignment":
        return "evidence_collection"

    if prediction == "strong_sred":
        return "claim_development"

    if prediction in {"borderline", "needs_more_info"}:
        return "technical_interview"

    if prediction == "routine" and uncertainty_signals:
        return "routine_exception_review"

    if prediction == "routine":
        return "routine_screening"

    if confidence < 0.45 or missing_checks or partial_checks or routine_signals:
        return "technical_interview"

    return "triage"


def _select_priority(prediction, confidence, case_stage):
    if case_stage in {"evidence_collection", "claim_development"}:
        if confidence >= 0.55:
            return "high"
        return "medium"

    if case_stage == "technical_interview":
        return "medium"

    if case_stage == "routine_exception_review":
        return "medium"

    if prediction == "routine" and confidence >= 0.55:
        return "low"

    return "low"


def _select_decision(case_stage, prediction):
    decisions = {
        "evidence_collection": "Proceed to evidence collection before claim drafting.",
        "claim_development": "Develop the SR&ED theory and close evidence gaps before treating this as claim-ready.",
        "technical_interview": "Run a technical interview to clarify uncertainty, experiments, results, and advancement.",
        "routine_exception_review": "Screen for hidden technological uncertainty before excluding the work as routine.",
        "routine_screening": "Treat as likely routine unless the client identifies failed standard methods or experimentation.",
        "triage": "Keep in triage until the technical facts are clearer.",
    }

    return decisions.get(
        case_stage,
        f"Continue triage for the {prediction} classification.",
    )


def _build_blockers(missing_checks, partial_checks, routine_signals):
    blockers = []

    for check_name in missing_checks:
        blockers.append(f"Missing CRA checklist support: {check_name}.")

    for check_name in partial_checks:
        blockers.append(f"Partial CRA checklist support: {check_name}.")

    if routine_signals:
        blockers.append(
            "Routine implementation signals need review: "
            + ", ".join(routine_signals)
            + "."
        )

    if not blockers:
        blockers.append("No major blockers detected, but supporting evidence should still be collected.")

    return blockers


def _prioritize_evidence_requests(evidence_map, checks, prediction):
    requests = []

    if checks["technological_uncertainty"]["status"] != "present":
        requests.append("A concise statement of the technological uncertainty and why standard practice was insufficient.")

    if checks["systematic_investigation"]["status"] != "present":
        requests.append("A timeline of hypotheses, prototypes, experiments, analyses, or iterations.")

    if checks["experimental_results"]["status"] != "present":
        requests.append("Measured results for each attempted approach, including failures and abandoned paths.")

    if checks["supporting_evidence"]["status"] != "present":
        requests.extend(evidence_map["missing_evidence"][:5])

    if prediction == "strong_sred":
        requests.append("A draft technical narrative connecting uncertainty, tests, results, and advancement.")

    return _unique_items(requests)


def _prioritize_questions(questions, cra_reference_questions, missing_checks):
    prioritized = []
    question_by_check = {
        "technological_advancement": "What scientific knowledge or technological capability was the team trying to advance?",
        "technological_uncertainty": "What scientific or technological uncertainty could not be resolved using existing knowledge or standard practice?",
        "systematic_investigation": "What systematic investigation or search was carried out by experiment or analysis?",
        "experimental_results": "What results or technical learning came from the tests, including failed attempts?",
        "supporting_evidence": "What records support the work, such as tickets, commits, test logs, benchmark results, design notes, or experiment records?",
    }

    for check_name in missing_checks:
        question = question_by_check.get(check_name)
        if question:
            prioritized.append(question)

    prioritized.extend(questions)
    prioritized.extend(cra_reference_questions)

    return _unique_items(prioritized)[:6]


def _build_action_plan(case_stage, blockers, evidence_requests, interview_focus):
    if case_stage in {"routine_screening", "routine_exception_review"}:
        return [
            "Confirm whether the work used only standard tools, vendor documentation, or known implementation patterns.",
            "Ask whether any technical limitation could not be resolved using standard practice.",
            "If no uncertainty or systematic investigation is identified, keep the work outside the SR&ED candidate pool.",
        ]

    plan = [
        "Start with the highest-priority interview questions.",
        "Collect the evidence items needed to close the main blockers.",
        "Update the project description with specific uncertainty, experiments, results, and learning.",
    ]

    if blockers:
        plan.append("Re-run the analyzer after the blockers are clarified.")

    if evidence_requests:
        plan.append("Attach the strongest supporting records before preparing a claim narrative.")

    if not interview_focus:
        plan.append("Prepare a concise reviewer handoff summary.")

    return plan


def _build_handoff_summary(
    prediction,
    confidence,
    case_stage,
    alignment,
    blockers,
    explanation,
):
    blocker_summary = blockers[0] if blockers else "No major blocker detected."
    reason_summary = explanation["reasons"][0] if explanation["reasons"] else "No reason summary available."

    return (
        f"Working classification is {prediction} with {confidence:.2f} confidence. "
        f"Case stage is {case_stage}; CRA alignment is {alignment}. "
        f"{reason_summary} Primary blocker: {blocker_summary}"
    )


def _unique_items(items):
    seen = set()
    unique = []

    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique

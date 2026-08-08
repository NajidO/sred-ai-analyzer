def check_against_cra_guidelines(text, signals):
    text_lower = text.lower()

    checks = {
        "technological_advancement": check_advancement(text_lower),
        "technological_uncertainty": check_uncertainty(text_lower, signals),
        "systematic_investigation": check_systematic_investigation(text_lower),
        "experimental_results": check_results_or_learning(text_lower),
        "supporting_evidence": check_supporting_evidence(text_lower),
    }

    score = sum(1 for result in checks.values() if result["status"] == "present")
    partial_score = sum(1 for result in checks.values() if result["status"] == "partial")

    if score >= 4:
        overall = "strong_alignment"
    elif score >= 2 or (score >= 1 and partial_score >= 2):
        overall = "partial_alignment"
    else:
        overall = "weak_alignment"

    return {
        "overall_alignment": overall,
        "checks": checks,
    }


def check_advancement(text_lower):
    keywords = [
        "new knowledge",
        "technological advancement",
        "advance",
        "improve understanding",
        "learned",
        "technical knowledge",
        "new approach",
        "custom method",
        "novel",
    ]

    if any(keyword in text_lower for keyword in keywords):
        return {
            "status": "present",
            "comment": "The description suggests possible technological advancement or new technical knowledge."
        }

    return {
        "status": "missing",
        "comment": "The description does not clearly explain what new technological knowledge or advancement was sought."
    }


def check_uncertainty(text_lower, signals):
    uncertainty_keywords = [
        "uncertainty",
        "unknown",
        "could not determine",
        "failed",
        "insufficient",
        "unpredictable",
        "not known",
        "could not achieve",
    ]

    if signals["uncertainty_signals"] or any(keyword in text_lower for keyword in uncertainty_keywords):
        return {
            "status": "present",
            "comment": "The description includes possible technological uncertainty indicators."
        }

    return {
        "status": "missing",
        "comment": "The description does not clearly state the scientific or technological uncertainty."
    }


def check_systematic_investigation(text_lower):
    strong_keywords = [
        "experiment",
        "experiments",
        "tested",
        "testing",
        "analysis",
        "prototype",
        "iteration",
        "benchmark",
        "evaluated",
        "compared",
        "trial",
        "hypothesis",
    ]

    if any(keyword in text_lower for keyword in strong_keywords):
        return {
            "status": "present",
            "comment": "The description suggests systematic investigation through testing, analysis, or experimentation."
        }

    return {
        "status": "missing",
        "comment": "The description does not clearly describe experiments, analysis, prototypes, or iterations."
    }


def check_results_or_learning(text_lower):
    learning_keywords = [
        "result",
        "results",
        "learned",
        "learning",
        "determined",
        "found that",
        "demonstrated",
        "showed",
        "failed",
        "improved",
        "reduced",
        "increased",
        "measured",
    ]

    if any(keyword in text_lower for keyword in learning_keywords):
        return {
            "status": "partial",
            "comment": "The description hints at results or learning, but specific measured outcomes should be documented."
        }

    return {
        "status": "missing",
        "comment": "The description does not clearly identify the results or technical knowledge gained."
    }


def check_supporting_evidence(text_lower):
    strong_evidence_keywords = [
        "jira",
        "ticket",
        "tickets",
        "git",
        "commit",
        "commits",
        "pull request",
        "pull requests",
        "test log",
        "test logs",
        "benchmark report",
        "benchmark results",
        "design document",
        "design documents",
        "technical document",
        "technical documents",
        "architecture diagram",
        "architecture diagrams",
        "experiment record",
        "experiment records",
        "sprint note",
        "sprint notes",
        "lab notes",
        "test results",
        "qa results",
        "performance report",
        "accuracy report",
    ]

    weak_evidence_keywords = [
        "benchmark",
        "measured",
        "compared accuracy",
        "accuracy results",
        "results",
    ]

    if any(keyword in text_lower for keyword in strong_evidence_keywords):
        return {
            "status": "present",
            "comment": "The description mentions possible supporting documentation or records."
        }

    if any(keyword in text_lower for keyword in weak_evidence_keywords):
        return {
            "status": "partial",
            "comment": "The description mentions results or measurements, but specific supporting documents should still be requested."
        }

    return {
        "status": "missing",
        "comment": "No supporting documentation is mentioned. Ask for Jira tickets, commits, test logs, benchmark results, design notes, and records of failed approaches."
    }
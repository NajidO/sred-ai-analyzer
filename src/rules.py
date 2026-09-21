UNCERTAINTY_PHRASES = [
    "failed",
    "failure",
    "standard methods failed",
    "existing methods failed",
    "known methods were insufficient",
    "insufficient",
    "could not determine",
    "unknown whether",
    "technical uncertainty",
    "unpredictable",
    "failed under load",
    "failed under high",
    "latency",
    "scalability",
    "data inconsistency",
    "accuracy threshold",
    "prototype",
    "tested multiple",
    "tested several",
    "experimented",
    "benchmark",
    "iteration",
    "hypothesis",
    "model architectures",
    "preprocessing",
]

ROUTINE_PHRASES = [
    "standard tools",
    "configured",
    "installed",
    "migrated",
    "dashboard",
    "updated the website",
    "bug",
    "standard integration",
    "off-the-shelf",
    "routine",
    "created reports",
]

RESOLVED_STANDARD_PHRASES = [
    "provided the required configuration",
    "worked because the environment matched",
    "matched its documented assumptions",
    "documented setting was selected",
    "standard template worked",
]


def extract_signals(text):
    text_lower = text.lower()

    uncertainty_hits = [
        phrase for phrase in UNCERTAINTY_PHRASES
        if phrase in text_lower
    ]

    routine_hits = [
        phrase for phrase in ROUTINE_PHRASES
        if phrase in text_lower
    ]

    return {
        "uncertainty_signals": uncertainty_hits,
        "routine_signals": routine_hits,
        "resolved_standard_configuration": is_resolved_standard_configuration(text),
    }


def is_resolved_standard_configuration(text):
    text_lower = " ".join(text.lower().split())
    routine_hits = [
        phrase
        for phrase in ROUTINE_PHRASES
        if phrase in text_lower
    ]
    return (
        len(routine_hits) >= 2
        and any(phrase in text_lower for phrase in RESOLVED_STANDARD_PHRASES)
    )


def apply_routine_prediction_guardrail(prediction, probabilities, labels, signals):
    if not signals.get("resolved_standard_configuration"):
        return prediction, probabilities

    labels_list = list(labels)
    if "routine" not in labels_list:
        return prediction, probabilities

    guarded = probabilities.copy()
    routine_index = labels_list.index("routine")
    routine_probability = max(float(max(guarded)), 0.76)
    other_total = sum(
        float(value)
        for index, value in enumerate(guarded)
        if index != routine_index
    )
    remaining = max(1.0 - routine_probability, 0.0)

    for index, value in enumerate(guarded):
        if index == routine_index:
            guarded[index] = routine_probability
        elif other_total:
            guarded[index] = float(value) / other_total * remaining
        else:
            guarded[index] = 0.0

    return "routine", guarded

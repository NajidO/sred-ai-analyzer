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
    }
CRA_ELIGIBILITY_POINTS = {
    "conducted_in_canada": {
        "question": "Was the work conducted in Canada?",
        "why_it_matters": (
            "CRA states that eligible SR&ED work must be conducted in Canada."
        ),
    },
    "advancement": {
        "question": "What scientific knowledge or technological capability was the team trying to advance?",
        "why_it_matters": (
            "CRA describes advancement as generating or discovering new knowledge "
            "that moves scientific or technological understanding forward."
        ),
    },
    "uncertainty": {
        "question": "What scientific or technological uncertainty could not be resolved using existing knowledge or standard practice?",
        "why_it_matters": (
            "CRA guidance explains that new knowledge is needed when it is unknown "
            "or uncertain whether a result can be achieved due to an insufficiency "
            "in the scientific or technological knowledge base."
        ),
    },
    "systematic_investigation": {
        "question": "What systematic investigation or search was carried out by experiment or analysis?",
        "why_it_matters": (
            "CRA states that systematic investigation is more than using a systematic "
            "approach. It should include defining a problem, advancing a hypothesis, "
            "testing by experiment or analysis, and developing logical conclusions."
        ),
    },
    "results_and_learning": {
        "question": "What results or technical learning came from the tests, including failed attempts?",
        "why_it_matters": (
            "CRA guidance notes that success is not required; learning that an approach "
            "does not work may still represent new scientific or technological knowledge."
        ),
    },
    "business_vs_technology": {
        "question": "Is the advancement technological, rather than only a business improvement?",
        "why_it_matters": (
            "CRA distinguishes advancement in science or technology from advancement "
            "of business practices or commercial outcomes."
        ),
    },
    "standard_practice": {
        "question": "Why could the issue not be resolved through standard methods, known techniques, vendor documentation, or available expertise?",
        "why_it_matters": (
            "CRA guidance emphasizes that the uncertainty should not be resolvable "
            "using reasonably available public or internal knowledge."
        ),
    },
    "supporting_evidence": {
        "question": "What records support the work, such as tickets, commits, test logs, benchmark results, design notes, or experiment records?",
        "why_it_matters": (
            "CRA technical review depends on clear descriptions and supporting facts "
            "showing the uncertainty, investigation, results, and advancement."
        ),
    },
}
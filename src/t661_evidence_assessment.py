import re


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?\s*"
    r"(?:nm(?:/minute)?|nm/min(?:ute)?|kv|mv|ma|amps?|v|%|minutes?|hours?|"
    r"seconds?|db|snr|ms|mb|gb|hz|khz|mhz|ghz)(?=$|[\s,.;:)])",
    re.IGNORECASE,
)

OBSERVED_OUTCOME_PHRASES = [
    "produced",
    "formed",
    "caused",
    "resulted",
    "observed",
    "showed",
    "reached",
    "achieved",
    "retained",
    "failed",
    "failure",
    "did not",
    "improved",
    "reduced",
    "increased",
    "removed",
    "delamination",
    "phase separation",
    "unbroken",
    "controlled",
    "mechanism",
]
EVIDENCE_PHRASES = [
    "recorded",
    "test log",
    "test logs",
    "data set",
    "dataset",
    "image sequence",
    "image sequences",
    "engineering note",
    "engineering notes",
    "test matrix",
    "test matrices",
    "design revision",
    "design revisions",
    "source commit",
    "source commits",
    "git commit",
    "git commits",
    "batch sheet",
    "batch sheets",
]


LINE_TITLES = {
    "242": "Technological uncertainties and limits of standard practice",
    "244": "Systematic investigation performed during the fiscal year",
    "246": "Technological advancements or knowledge gained",
}


def build_t661_evidence_assessment(source_text, source_sections, strategy):
    if not source_sections and source_text.strip():
        source_sections = {
            1: {"title": "Project objective and uncertainty", "text": source_text},
            5: {"title": "Approaches, tests, and results", "text": source_text},
            13: {"title": "Knowledge learned and advancement", "text": source_text},
        }

    line_242 = assess_line_242(source_text, source_sections, strategy["streams"])
    line_244 = assess_line_244(source_text, source_sections, strategy["streams"])
    line_246 = assess_line_246(source_text, source_sections, strategy["streams"])
    sections = {"242": line_242, "244": line_244, "246": line_246}
    consistency_issues = detect_consistency_issues(source_text)
    for issue in consistency_issues:
        for line_number in issue["lines"]:
            section = sections[line_number]
            if issue["message"] not in section["missing_information"]:
                section["missing_information"].append(issue["message"])
            section["status"] = "needs_more_information"

    routine_flags = detect_routine_only_work(source_text)
    blocked_lines = [
        line_number
        for line_number, section in sections.items()
        if section["status"] != "ready"
    ]
    can_draft = not blocked_lines and not routine_flags

    if routine_flags:
        decision = "not_report_ready"
        summary = (
            "Do not draft a T661 project description. The source indicates that documented "
            "vendor or standard configuration resolved the stated objective."
        )
    elif can_draft:
        decision = "draft_ready"
        summary = (
            "The available source supports drafting all three T661 project-description "
            "lines, subject to analyst verification."
        )
    else:
        decision = "needs_more_information"
        summary = (
            "Do not draft the T661 project description yet. Collect the missing "
            f"information for line(s) {', '.join(blocked_lines)} first."
        )

    return {
        "decision": decision,
        "can_draft": can_draft,
        "blocked_lines": blocked_lines,
        "summary": summary,
        "routine_flags": routine_flags,
        "consistency_issues": consistency_issues,
        "sections": sections,
    }


def assess_line_242(source_text, source_sections, streams):
    relevant_text = collect_section_text(
        source_sections,
        fallback_numbers=range(1, 5),
        title_terms=[
            "achieve",
            "develop",
            "improve",
            "overview",
            "objective",
            "technology",
            "challenge",
            "difficulty",
            "uncertainty",
            "standard",
        ],
    )
    checks = [
        {
            "id": "quantified_target",
            "present": (
                bool(MEASUREMENT_PATTERN.search(relevant_text))
                and contains_any(
                    relevant_text,
                    [
                        "target",
                        "objective",
                        "required",
                        "maintain",
                        "sought",
                        "needed",
                        "had to",
                        "below",
                        "less than",
                    ],
                )
            )
            or contains_any(
                relevant_text,
                [
                    "acceptance required",
                    "acceptance criteria",
                    "accepted only if",
                    "required an unbroken",
                    "required a continuous",
                ],
            ),
            "missing": (
                "Quantified baseline, required technical target, and acceptance threshold "
                "for each uncertainty."
            ),
        },
        {
            "id": "technological_unknown",
            "present": contains_any(
                relevant_text,
                [
                    "uncertain",
                    "uncertainty",
                    "unsure",
                    "whether",
                    "unknown",
                    "could not predict",
                    "did not know",
                ],
            ),
            "missing": (
                "The exact technological relationship or capability that could not be "
                "predicted using the available knowledge."
            ),
        },
        {
            "id": "standard_practice_limit",
            "present": contains_any(
                relevant_text,
                [
                    "did not provide",
                    "did not accurately",
                    "could not",
                    "insufficient",
                    "failed",
                    "not simply",
                    "did not work",
                ],
            )
            and contains_any(
                relevant_text,
                [
                    "standard",
                    "existing",
                    "published",
                    "supplier",
                    "known",
                    "simulation",
                    "calculation",
                ],
            ),
            "missing": (
                "The known methods, calculations, supplier guidance, or prior designs tried "
                "and the observed reason each could not resolve the uncertainty."
            ),
        },
    ]
    return build_section_assessment(
        "242",
        checks,
        source_sections,
        streams,
        fallback_numbers=range(1, 5),
        title_terms=[
            "achieve",
            "develop",
            "improve",
            "overview",
            "objective",
            "technology",
            "challenge",
            "difficulty",
            "uncertainty",
            "standard",
        ],
        question_builder=build_line_242_questions,
    )


def assess_line_244(source_text, source_sections, streams):
    relevant_text = collect_section_text(
        source_sections,
        fallback_numbers=range(5, 13),
        title_terms=[
            "approach",
            "work",
            "test",
            "experiment",
            "result",
            "next",
            "hypothesis",
            "hypoth",
            "record",
            "theory",
            "trial",
            "sequence",
            "changed",
            "pivot",
            "evidence",
            "archive",
            "configuration",
        ],
    )
    checks = [
        {
            "id": "hypothesis_or_expected_result",
            "present": contains_any(
                relevant_text,
                ["hypothesis", "expected", "we believed", "intended to determine"],
            ),
            "missing": (
                "The hypothesis or expected technical result for each SIS iteration."
            ),
        },
        {
            "id": "alternatives_and_variables",
            "present": has_tested_alternatives(relevant_text),
            "missing": (
                "The alternatives tested or considered, controlled variables, test conditions, "
                "and reason each alternative was selected."
            ),
        },
        {
            "id": "observed_results",
            "present": bool(MEASUREMENT_PATTERN.search(relevant_text))
            or contains_any(relevant_text, OBSERVED_OUTCOME_PHRASES),
            "missing": (
                "Observed or measured results for every material iteration, including the "
                "baseline, target or acceptance criteria, test conditions, and outcome."
            ),
        },
        {
            "id": "failure_and_decision",
            "present": has_failure_and_decision(relevant_text),
            "missing": (
                "For each result, why the approach failed or only partly worked and whether it "
                "was refined, abandoned, or caused a pivot to the next experiment."
            ),
        },
        {
            "id": "supporting_records",
            "present": contains_non_negated_phrase(relevant_text, EVIDENCE_PHRASES),
            "missing": (
                "The contemporaneous records supporting the SIS sequence, such as test matrices, "
                "logs, images, simulations, design revisions, and engineering notes."
            ),
        },
    ]
    return build_section_assessment(
        "244",
        checks,
        source_sections,
        streams,
        fallback_numbers=range(5, 13),
        title_terms=[
            "approach",
            "work",
            "test",
            "experiment",
            "result",
            "next",
            "hypothesis",
            "hypoth",
            "record",
            "theory",
            "trial",
            "sequence",
            "changed",
            "pivot",
            "evidence",
            "archive",
            "configuration",
        ],
        question_builder=build_line_244_questions,
    )


def assess_line_246(source_text, source_sections, streams):
    relevant_text = collect_section_text(
        source_sections,
        fallback_numbers=range(13, 17),
        title_terms=[
            "learn",
            "knowledge",
            "advancement",
            "improvement",
            "failed",
            "finding",
            "outcome",
            "boundary",
            "remaining",
            "remained",
            "open",
        ],
    )
    checks = [
        {
            "id": "explicit_technical_learning",
            "present": contains_any(
                relevant_text,
                ["determined", "established", "learned", "gained knowledge", "showed"],
            ),
            "missing": (
                "The technological knowledge established for each TU, stated separately from "
                "product or business benefits."
            ),
        },
        {
            "id": "causal_learning_from_results",
            "present": contains_any(
                relevant_text,
                [
                    "because",
                    "not viable",
                    "insufficient",
                    "prevented",
                    "mechanism",
                    "relationship",
                ],
            ),
            "missing": (
                "How the observed results and failed alternatives changed the team's technical "
                "understanding for each TU."
            ),
        },
        {
            "id": "advancement_boundaries",
            "present": bool(MEASUREMENT_PATTERN.search(relevant_text))
            or (
                contains_any(
                    relevant_text,
                    ["established", "determined", "retained", "demonstrated"],
                )
                and contains_any(relevant_text, OBSERVED_OUTCOME_PHRASES)
            ),
            "missing": (
                "The capability or technical boundary established by the work, including the "
                "conditions where the observed result did and did not hold."
            ),
        },
        {
            "id": "remaining_uncertainty",
            "present": contains_any(
                relevant_text,
                [
                    "remained technologically uncertain",
                    "remained unresolved",
                    "still unknown",
                    "not fully resolved",
                    "remained problematic",
                ],
            ),
            "missing": (
                "The technological uncertainty that remained at fiscal year-end and the boundary "
                "between knowledge gained and work still required."
            ),
        },
    ]
    return build_section_assessment(
        "246",
        checks,
        source_sections,
        streams,
        fallback_numbers=range(13, 17),
        title_terms=[
            "learn",
            "knowledge",
            "advancement",
            "improvement",
            "failed",
            "finding",
            "outcome",
            "boundary",
            "remaining",
            "remained",
            "open",
        ],
        question_builder=build_line_246_questions,
    )


def build_section_assessment(
    line_number,
    checks,
    source_sections,
    streams,
    fallback_numbers,
    title_terms,
    question_builder,
):
    relevant_sentences = collect_section_sentences(
        source_sections,
        fallback_numbers,
        title_terms,
    )
    missing_information = [check["missing"] for check in checks if not check["present"]]
    status = "ready" if not missing_information else "needs_more_information"
    stream_assessments = []

    for index, stream in enumerate(streams, start=1):
        facts = select_stream_facts(relevant_sentences, stream)
        questions = [] if status == "ready" else question_builder(stream, index)
        stream_assessments.append({
            "stream_id": stream["id"],
            "sis_id": f"SIS{index}",
            "title": stream["title"],
            "supported_information": facts,
            "questions": questions,
        })

    return {
        "line": line_number,
        "title": LINE_TITLES[line_number],
        "status": status,
        "supported_information": unique_items(relevant_sentences)[:15],
        "missing_information": missing_information,
        "checks": checks,
        "stream_assessments": stream_assessments,
    }


def build_line_242_questions(stream, index):
    topics = format_stream_topics(stream)
    return [
        (
            f"For {stream['id']} ({stream['title']}), what exact technical outcome or "
            f"relationship among {topics} could not be predicted before experimentation, and "
            "under which operating conditions?"
        ),
        (
            f"For {stream['id']}, which known designs, calculations, supplier guidance, or "
            "standard methods were tried, and what observed result showed that each was insufficient?"
        ),
        (
            f"For {stream['id']}, what was the starting capability, required target, and "
            "acceptance threshold, including the metric and test conditions?"
        ),
    ]


def build_line_244_questions(stream, index):
    topics = format_stream_topics(stream)
    sis_id = f"SIS{index}"
    return [
        (
            f"For {sis_id}/{stream['id']} ({stream['title']}), which alternatives involving "
            f"{topics} were tested, and which were considered but not tested?"
        ),
        (
            f"For each {sis_id}/{stream['id']} alternative, what result was expected, what "
            "variables and conditions were controlled, and what was actually measured?"
        ),
        (
            f"Which {sis_id}/{stream['id']} alternatives failed or only partly met the target, "
            "why were they insufficient, and was each one refined, abandoned, or replaced by a pivot?"
        ),
        (
            f"What further {sis_id}/{stream['id']} experimentation followed each observation, "
            "and what evidence records the sequence and decision to continue or stop?"
        ),
    ]


def build_line_246_questions(stream, index):
    sis_id = f"SIS{index}"
    return [
        (
            f"What technological knowledge did {sis_id}/{stream['id']} establish that was not "
            "available from the starting methods or published guidance?"
        ),
        (
            f"What did the failed or abandoned {sis_id}/{stream['id']} alternatives reveal about "
            "the underlying technical relationships or limitations?"
        ),
        (
            f"Under which measured conditions did the {sis_id}/{stream['id']} learning hold, "
            "where did it stop holding, and what remained uncertain at fiscal year-end?"
        ),
    ]


def collect_section_text(source_sections, fallback_numbers, title_terms):
    sections = select_sections(source_sections, fallback_numbers, title_terms)
    return " ".join(
        " ".join(section["text"].split())
        for section in sections
    ).lower()


def collect_section_sentences(source_sections, fallback_numbers, title_terms):
    sections = select_sections(source_sections, fallback_numbers, title_terms)
    sentences = []
    for section in sections:
        sentences.extend(split_sentences(section["text"]))
    return unique_items(sentences)


def select_sections(source_sections, fallback_numbers, title_terms):
    fallback_numbers = set(fallback_numbers)
    title_matches = [
        section
        for _, section in sorted(source_sections.items())
        if any(term in section["title"].lower() for term in title_terms)
    ]
    if title_matches:
        return title_matches

    return [
        section
        for number, section in sorted(source_sections.items())
        if number in fallback_numbers
    ]


def select_stream_facts(sentences, stream):
    terms = list(stream.get("evidence_terms", []))
    terms.extend(title_keywords(stream.get("title", "")))
    normalized_terms = unique_items(term.lower() for term in terms if len(term) > 2)
    facts = [
        sentence
        for sentence in sentences
        if any(term in sentence.lower() for term in normalized_terms)
    ]
    return unique_items(facts)[:5]


def title_keywords(title):
    stop_words = {
        "and",
        "the",
        "with",
        "from",
        "into",
        "under",
        "within",
        "whether",
    }
    words = re.findall(r"[a-z0-9-]+", title.lower())
    return [word for word in words if len(word) > 3 and word not in stop_words]


def format_stream_topics(stream):
    topics = unique_items(stream.get("evidence_terms", []))[:5]
    if not topics:
        return "the identified technical factors"
    if len(topics) == 1:
        return topics[0]
    return ", ".join(topics[:-1]) + f", and {topics[-1]}"


def split_sentences(text):
    normalized = " ".join(text.replace("**", "").split())
    if not normalized:
        return []
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(normalized) if sentence.strip()]


def contains_any(text, phrases):
    normalized = text.lower()
    return any(phrase in normalized for phrase in phrases)


def contains_non_negated_phrase(text, phrases):
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        normalized = sentence.lower()
        unavailable_record = re.search(
            r"\b(?:files?|records?|logs?|notes?|commits?|versions?|matrices|data(?:sets?)?)\b"
            r"[^.!?]{0,80}\b(?:unavailable|not available|cannot be produced|could not be "
            r"produced|cannot be tied|could not be tied|not retained|not accessible)\b",
            normalized,
        )
        for phrase in phrases:
            for match in re.finditer(re.escape(phrase), normalized):
                start, end = match.span()
                prefix = normalized[max(0, start - 55):start]
                suffix = normalized[end:]
                prefix_negated = re.search(
                    r"(?:^|\b)(?:no|none|without|missing|unavailable)\b[^.!?]{0,45}$",
                    prefix,
                )
                suffix_negated = re.search(
                    r"\b(?:were not|was not|not recorded|not available|unavailable|"
                    r"do not exist|cannot be identified|could not be identified|"
                    r"cannot be produced|could not be produced|cannot be tied|"
                    r"could not be tied|not retained|not accessible)\b",
                    suffix,
                )
                if not prefix_negated and not suffix_negated and not unavailable_record:
                    return True

    return False


def has_tested_alternatives(text):
    test_actions = [
        "tested",
        "built",
        "produced",
        "tried",
        "investigated",
        "compared",
        "evaluated",
    ]
    if not contains_any(text, test_actions):
        return False

    explicit_alternatives = [
        "combinations",
        "variants",
        "multiple",
        "several",
        "different",
        "alternatives",
        "configurations",
    ]
    if contains_any(text, explicit_alternatives):
        return True

    count_word = (
        r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
    )
    enumerated_design = re.compile(
        rf"\b{count_word}\s+(?:[a-z][a-z0-9-]*\s+){{0,3}}"
        r"(?:models?|filters?|intervals?|methods?|designs?|geometr(?:y|ies)|sequences?|"
        r"approaches?|algorithms?|prototypes?|formulations?|settings?|stages?|windows?|"
        r"controllers?|inserts?)\b",
        re.IGNORECASE,
    )
    return bool(enumerated_design.search(text))


def has_failure_and_decision(text):
    failure_markers = [
        "failed",
        "rejected",
        "not viable",
        "did not",
        "not consistently",
        "too late",
        "missed the target",
        "exceeded",
        "caused",
        "destabilized",
        "increased",
        "problematic",
        "insufficient",
        "worsened",
    ]
    decision_markers = [
        "then",
        "next",
        "after",
        "abandoned",
        "pivot",
        "refined",
        "replaced",
        "selected",
        "advanced",
        "revised",
    ]
    return contains_any(text, failure_markers) and contains_any(text, decision_markers)


def detect_consistency_issues(source_text):
    normalized = " ".join(source_text.lower().split())
    issues = []

    if contains_any(
        normalized,
        [
            "another section states",
            "a later note states",
            "conflicting statement",
            "contradicts the earlier",
        ],
    ):
        issues.append({
            "lines": ["242"],
            "message": (
                "The source contains conflicting statements about the starting technology "
                "or whether standard practice resolved the problem."
            ),
        })

    if contains_any(
        normalized,
        [
            "no experiments were actually run",
            "no tests were actually run",
            "all values were projections",
            "no measured output was retained",
            "results were estimated rather than measured",
        ],
    ):
        issues.append({
            "lines": ["244", "246"],
            "message": (
                "The source describes projected or unverified results and also states that "
                "experiments or measured outputs are unavailable."
            ),
        })

    fiscal_year_denial = re.search(
        r"\bno\s+(?:experimental work|experiments?|testing|tests)\s+"
        r"(?:actually\s+)?(?:occurred|took place|was performed|were performed|"
        r"was conducted|were conducted)\b",
        normalized,
    )
    if fiscal_year_denial:
        issues.append({
            "lines": ["244"],
            "message": (
                "The source states that no experimental work occurred in the claimed period; "
                "confirm the fiscal-year chronology before drafting Line 244."
            ),
        })

    return issues


def detect_routine_only_work(source_text):
    normalized = " ".join(source_text.lower().split())
    routine_indicators = [
        "off-the-shelf",
        "vendor's documented",
        "vendor documentation provided",
        "documented standard template",
        "documented high-availability template",
        "standard configuration",
    ]
    resolved_by_standard_indicators = [
        "provided the required configuration",
        "worked because the environment matched",
        "matched its documented assumptions",
        "documented setting was selected",
        "standard template worked",
    ]

    if (
        sum(indicator in normalized for indicator in routine_indicators) >= 2
        and contains_any(normalized, resolved_by_standard_indicators)
    ):
        return [
            "The stated objective was resolved by documented vendor or standard configuration."
        ]

    return []


def unique_items(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result

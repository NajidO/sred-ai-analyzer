import re


STREAM_DEFINITIONS = [
    {
        "id": "TU1",
        "title": "Electron beam resolution and focusing in the shortened column",
        "keywords": [
            "spot size",
            "resolution",
            "focusing",
            "objective",
            "pole-piece",
            "pole piece",
            "lens",
            "aperture",
            "aberration",
        ],
        "uncertainty": (
            "Whether the shortened electron-optical geometry could maintain the "
            "required beam spot size, focusing repeatability, and useful imaging resolution."
        ),
        "investigation": (
            "Pole-piece geometry, lens-current, working-distance, aperture, and simulation/prototype tests."
        ),
        "question_templates": [
            (
                "For TU1, can you provide the test matrix for pole-piece variants, "
                "lens currents, working distances, aperture sizes, and accelerating voltages?"
            ),
            (
                "For TU1, which configurations improved resolution but failed on focus "
                "repeatability, sample-height tolerance, or working-distance range?"
            ),
            (
                "For TU1, what alternatives were considered besides reducing the pole-piece "
                "gap and increasing objective-lens excitation?"
            ),
        ],
    },
    {
        "id": "TU2",
        "title": "Thermal drift, magnetic history, and active compensation",
        "keywords": [
            "thermal",
            "temperature",
            "drift",
            "hysteresis",
            "history",
            "compensation",
            "focus correction",
            "beam displacement",
            "stabilization",
        ],
        "uncertainty": (
            "Whether thermal state, magnetic history, and software compensation could control "
            "focus shift and image drift within the compact architecture."
        ),
        "investigation": (
            "Thermal instrumentation, passive thermal experiments, model-based compensation, "
            "and image-based drift correction."
        ),
        "question_templates": [
            (
                "For TU2, can you provide temperature logs, lens-current histories, and "
                "drift measurements before and after each compensation approach?"
            ),
            (
                "For TU2, did the temperature-based correction assume a linear relationship? "
                "If so, where did that assumption fail, and did any operating sequence suggest "
                "that lens-current or magnetic history mattered?"
            ),
            (
                "For TU2, was image-based drift correction tested or considered? If so, how did "
                "it perform on low-contrast samples, and what alternatives were evaluated?"
            ),
        ],
    },
    {
        "id": "TU3",
        "title": "Secondary-electron detector signal and field interaction",
        "keywords": [
            "detector",
            "snr",
            "signal-to-noise",
            "secondary-electron",
            "bias",
            "shielding",
            "collection",
            "low beam current",
            "charging",
        ],
        "uncertainty": (
            "Whether detector geometry, biasing, and shielding could improve signal collection "
            "without disturbing the beam near the sample."
        ),
        "investigation": (
            "Detector placement, segmented annular detector tests, detector bias settings, "
            "shielding revisions, and SNR measurements."
        ),
        "question_templates": [
            (
                "For TU3, can you provide SNR results by detector geometry, detector distance, "
                "bias voltage, beam current, and sample type?"
            ),
            (
                "For TU3, which detector positions increased signal but caused non-uniformity, "
                "charging, or beam disturbance near the sample?"
            ),
            (
                "For TU3, what shielding or bias alternatives were tested before selecting "
                "the revised detector configuration?"
            ),
        ],
    },
    {
        "id": "TU4",
        "title": "Software performance, accuracy, or reliability under non-routine constraints",
        "keywords": [
            "machine learning",
            "ml model",
            "recommendation",
            "software scalability",
            "classifier",
            "latency",
            "database",
            "api timeout",
            "queue",
            "prediction",
            "accuracy",
            "scalability",
            "throughput",
            "software",
            "model",
            "classification",
            "recall",
            "false positive",
            "inference",
            "embedding",
            "feature",
            "algorithm",
            "batching",
            "event-driven",
            "scheduling",
            "sequence",
            "temporal",
        ],
        "uncertainty": (
            "Whether software, data-processing, or algorithmic methods could meet the required "
            "performance, accuracy, reliability, or scalability target."
        ),
        "investigation": (
            "Algorithm, architecture, data-processing, benchmark, tuning, or simulation tests."
        ),
        "question_templates": [
            (
                "For TU4, what baseline method failed, what metric missed target, and what "
                "technical constraint made standard implementation insufficient?"
            ),
            (
                "For TU4, which algorithmic or architectural alternatives were tested, and "
                "what benchmark data showed each result?"
            ),
            (
                "For TU4, what edge cases or operating conditions still failed after the best approach?"
            ),
        ],
    },
]


def build_report_strategy(case_data, final_assessment, source_sections):
    source_text = case_data.get("updated_text", "")
    streams = identify_streams(source_text, source_sections)
    selected_structure = select_structure(streams, source_sections)
    rationale = build_strategy_rationale(streams, selected_structure, final_assessment, source_sections)
    questions = build_strategy_questions(streams, final_assessment)

    return {
        "selected_structure": selected_structure,
        "rationale": rationale,
        "streams": streams,
        "specific_questions": questions,
    }


def identify_streams(source_text, source_sections):
    normalized_text = source_text.lower()
    streams = []

    for definition in STREAM_DEFINITIONS:
        evidence = collect_keyword_evidence(
            definition["keywords"],
            normalized_text,
            source_sections,
        )

        if evidence["score"] >= 2:
            streams.append({
                "id": definition["id"],
                "title": definition["title"],
                "uncertainty": definition["uncertainty"],
                "investigation": definition["investigation"],
                "evidence_terms": evidence["terms"],
                "source_questions": evidence["source_questions"],
                "suggested_questions": definition["question_templates"],
            })

    uncertainty_clauses = extract_uncertainty_clauses(source_text, source_sections)
    uncovered_clauses = [
        clause
        for clause in uncertainty_clauses
        if not clause_matches_stream(clause, streams)
    ]

    for clause in uncovered_clauses:
        streams.append(
            build_source_defined_stream(
                source_text,
                source_sections,
                uncertainty_sentence=clause,
                stream_id=next_available_stream_id(streams),
            )
        )

    if not streams:
        streams.append(
            build_source_defined_stream(
                source_text,
                source_sections,
                stream_id="TU1",
            )
        )

    return streams


def collect_keyword_evidence(keywords, normalized_text, source_sections):
    terms = [
        keyword
        for keyword in keywords
        if has_keyword(normalized_text, keyword)
    ]
    source_questions = []

    for number, section in sorted(source_sections.items()):
        section_text = section["text"].lower()
        if any(has_keyword(section_text, keyword) for keyword in keywords):
            source_questions.append(f"Q{number}: {section['title']}")

    score = len(terms)

    return {
        "score": score,
        "terms": terms[:8],
        "source_questions": source_questions[:6],
    }


def build_source_defined_stream(
    source_text,
    source_sections,
    uncertainty_sentence="",
    stream_id="TU1",
):
    uncertainty_sentence = uncertainty_sentence or find_source_sentence(
        source_text,
        UNCERTAINTY_PHRASES,
    )
    investigation_sentence = find_source_sentence(
        source_text,
        [
            "tested",
            "compared",
            "experiment",
            "prototype",
            "trial",
        ],
    )

    if uncertainty_sentence:
        title = build_source_defined_title(uncertainty_sentence)
        uncertainty = uncertainty_sentence
    else:
        title = "Source-defined technological uncertainty"
        uncertainty = (
            "The source suggests a possible technological uncertainty, but the exact "
            "unknown still requires confirmation."
        )

    investigation = investigation_sentence or (
        "The source does not yet provide a sufficiently specific experiment or analysis sequence."
    )
    evidence_terms = extract_source_terms(uncertainty_sentence or source_text)
    source_questions = [
        f"Q{number}: {section['title']}"
        for number, section in sorted(source_sections.items())
        if contains_uncertainty_language(section["text"])
    ][:6]

    return {
        "id": stream_id,
        "title": title,
        "uncertainty": uncertainty,
        "investigation": investigation,
        "evidence_terms": evidence_terms,
        "source_questions": source_questions,
        "suggested_questions": [
            "What exact target could not be achieved with standard practice?",
            "Which standard approach failed first, and what result showed that it was insufficient?",
            "What alternatives were tested, abandoned, or kept for further work?",
        ],
    }


UNCERTAINTY_PHRASES = [
    "technological uncertainty",
    "technical uncertainty",
    "it was uncertain",
    "was uncertain",
    "did not know whether",
    "unknown whether",
    "uncertainty was whether",
]


def extract_uncertainty_clauses(source_text, source_sections):
    candidate_sentences = []
    uncertainty_title_terms = ["uncertaint", "unknown", "challenge", "difficulty"]

    for _, section in sorted(source_sections.items()):
        title = section.get("title", "").lower()
        section_sentences = [
            sentence
            for sentence in split_source_sentences(section.get("text", ""))
            if not is_remaining_uncertainty_statement(sentence)
        ]
        if any(term in title for term in uncertainty_title_terms):
            candidate_sentences.extend(section_sentences)
        else:
            candidate_sentences.extend(
                sentence
                for sentence in section_sentences
                if contains_uncertainty_language(sentence)
            )

    if not candidate_sentences:
        cleaned_source = re.sub(r"(?m)^\s*#+\s*[^\n]+$", " ", source_text)
        candidate_sentences = [
            sentence
            for sentence in split_source_sentences(cleaned_source)
            if contains_uncertainty_language(sentence)
            and not is_remaining_uncertainty_statement(sentence)
        ]

    clauses = []
    for sentence in candidate_sentences:
        whether_match = re.search(r"\bwhether\s+(.+)", sentence, re.IGNORECASE)
        if not whether_match:
            clauses.append(sentence.strip())
            continue

        whether_text = whether_match.group(1).strip()
        parts = re.split(r"\s+(?:or|and)\s+whether\s+", whether_text, flags=re.IGNORECASE)
        clauses.extend(f"Whether {part.strip()}" for part in parts if part.strip())

    return unique_items(clauses)


def is_remaining_uncertainty_statement(sentence):
    normalized = sentence.lower()
    return contains_uncertainty_language(normalized) and any(
        phrase in normalized
        for phrase in [
            "at fiscal year-end",
            "at year-end",
            "remained technologically uncertain",
            "remained unresolved",
            "remaining uncertainty",
            "still unknown at",
        ]
    )


def split_source_sentences(text):
    normalized = " ".join(text.replace("**", "").split())
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def clause_matches_stream(clause, streams):
    normalized_clause = clause.lower()
    for stream in streams:
        terms = list(stream.get("evidence_terms", []))
        terms.extend(stream_title_keywords(stream.get("title", "")))
        if any(has_keyword(normalized_clause, term.lower()) for term in terms):
            return True
    return False


def stream_title_keywords(title):
    stop_words = {
        "and",
        "from",
        "into",
        "source-defined",
        "the",
        "under",
        "whether",
        "with",
    }
    return [
        word
        for word in re.findall(r"[a-z][a-z0-9-]+", title.lower())
        if len(word) > 3 and word not in stop_words
    ]


def next_available_stream_id(streams):
    used = {stream["id"] for stream in streams}
    index = 1
    while f"TU{index}" in used:
        index += 1
    return f"TU{index}"


def find_source_sentence(source_text, phrases):
    normalized = " ".join(source_text.replace("**", "").split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in phrases):
            return sentence.strip()
    return ""


def build_source_defined_title(uncertainty_sentence):
    fragment = uncertainty_sentence
    prefixes = [
        "The team did not know whether ",
        "It was uncertain whether ",
        "The uncertainty was whether ",
        "The technological uncertainty was whether ",
        "The technical uncertainty was whether ",
    ]
    for prefix in prefixes:
        if fragment.lower().startswith(prefix.lower()):
            fragment = fragment[len(prefix):]
            break

    fragment = re.split(r"[.;]", fragment, maxsplit=1)[0].strip()
    words = fragment.split()
    if len(words) > 14:
        fragment = " ".join(words[:14]) + "..."
    if not fragment:
        return "Source-defined technological uncertainty"
    return "Source-defined: " + fragment[0].upper() + fragment[1:]


def extract_source_terms(text):
    stop_words = {
        "about",
        "after",
        "before",
        "could",
        "did",
        "from",
        "have",
        "into",
        "know",
        "might",
        "that",
        "team",
        "their",
        "there",
        "these",
        "they",
        "this",
        "under",
        "uncertain",
        "uncertainty",
        "whether",
        "with",
        "would",
    }
    terms = []
    for word in re.findall(r"[a-z][a-z0-9-]+", text.lower()):
        if len(word) < 5 or word in stop_words or word in terms:
            continue
        terms.append(word)
    return terms[:8]


def contains_uncertainty_language(text):
    normalized = text.lower()
    return any(
        phrase in normalized
        for phrase in [
            "uncertain",
            "uncertainty",
            "did not know",
            "unknown whether",
        ]
    )


def unique_items(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def has_keyword(text, keyword):
    if " " in keyword or "-" in keyword:
        return keyword in text

    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return keyword in tokens or f"{keyword}s" in tokens


def select_structure(streams, source_sections):
    if len(streams) >= 3:
        return {
            "mode": "split_by_uncertainty_stream",
            "label": "Split into TU/SIS streams",
            "summary": (
                "Use separate TU/SIS streams because the source facts describe multiple "
                "overlapping uncertainties and investigations."
            ),
        }

    if len(streams) == 2:
        return {
            "mode": "hybrid_t661_with_two_streams",
            "label": "Hybrid T661 narrative with two TU/SIS streams",
            "summary": (
                "Use a single T661 answer but organize line 244 and analyst notes around "
                "two technical streams."
            ),
        }

    if source_sections and len(source_sections) >= 10:
        return {
            "mode": "chronological_integrated",
            "label": "Integrated chronological T661 narrative",
            "summary": (
                "Use one integrated narrative because the project appears to have one main "
                "technical uncertainty with chronological experiments."
            ),
        }

    return {
        "mode": "intake_first",
        "label": "Intake-first narrative",
        "summary": (
            "Do not over-structure the report yet; collect more facts before deciding "
            "whether to split the claim."
        ),
    }


def build_strategy_rationale(streams, selected_structure, final_assessment, source_sections):
    rationale = [
        selected_structure["summary"],
        (
            f"Detected {len(streams)} candidate technical stream(s) from the source record."
        ),
    ]

    alignment = final_assessment.get("cra_alignment", "")
    if alignment:
        rationale.append(
            f"CRA checklist alignment is {alignment}, so the next drafting issue is structure and evidence quality."
        )

    if source_sections:
        rationale.append(
            f"The source has {len(source_sections)} numbered questionnaire sections, so the planner can use the client facts rather than only generic classifier signals."
        )

    if len(streams) >= 2:
        rationale.append(
            "Splitting helps avoid blending different experiments, failures, and learnings into one broad claim narrative."
        )

    return rationale


def build_strategy_questions(streams, final_assessment):
    questions = []

    for stream in streams:
        questions.extend(stream["suggested_questions"][:3])

    questions.extend(build_cross_cutting_questions(streams))

    agent_assessment = final_assessment.get("agent_assessment", {})
    for question in agent_assessment.get("interview_focus", []):
        if len(questions) >= 16:
            break

        if not is_generic_question(question):
            questions.append(question)

    return unique_items(questions)[:16]


def build_cross_cutting_questions(streams):
    stream_ids = {stream["id"] for stream in streams}
    questions = [
        (
            "What quantified baseline, target, and acceptance threshold applied to each "
            "technical stream, and which metric showed that standard practice was insufficient?"
        ),
        (
            "For each prototype or test, when did it occur within the fiscal year, what was "
            "changed, what was observed, and what decision led to the next iteration?"
        ),
        (
            "At fiscal year-end, which operating conditions or technical uncertainties remained "
            "unresolved, and which work continued into the following year?"
        ),
        (
            "Which records support each stream, such as design files, simulation versions, test "
            "matrices, logs, images, temperature or current histories, and engineering notes?"
        ),
    ]

    if {"TU1", "TU2", "TU3"}.issubset(stream_ids):
        questions.append(
            "What new technological knowledge was established separately for electron focusing, "
            "thermal or magnetic stability, and detector field interaction?"
        )

    return questions


def is_generic_question(question):
    normalized = question.lower()
    generic_phrases = [
        "what failed",
        "what experiments",
        "what was unknown",
        "what new technical knowledge",
        "what was the actual technical problem",
    ]
    return any(phrase in normalized for phrase in generic_phrases)


def unique_items(items):
    seen = set()
    unique = []

    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique

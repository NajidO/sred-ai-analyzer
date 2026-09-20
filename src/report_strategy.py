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

    if not streams:
        streams.append({
            "id": "TU1",
            "title": "Primary technological uncertainty",
            "uncertainty": (
                "The available information suggests a possible technological uncertainty, "
                "but the agent could not confidently split it into separate technical streams."
            ),
            "investigation": "Use an integrated T661 narrative until more detailed facts are available.",
            "evidence_terms": [],
            "source_questions": [],
            "suggested_questions": [
                "What exact target could not be achieved with standard practice?",
                "Which standard approach failed first, and what result showed that it was insufficient?",
                "What alternatives were tested, abandoned, or kept for further work?",
            ],
        })

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

    score = len(terms) + min(len(source_questions), 3)

    return {
        "score": score,
        "terms": terms[:8],
        "source_questions": source_questions[:6],
    }


def has_keyword(text, keyword):
    if " " in keyword or "-" in keyword:
        return keyword in text

    tokens = text.replace("/", " ").replace(",", " ").replace(".", " ").split()
    return keyword in tokens


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

from cra_reference import CRA_ELIGIBILITY_POINTS

def generate_followup_questions(prediction, signals):
    questions = []

    uncertainty_signals = signals["uncertainty_signals"]
    routine_signals = signals["routine_signals"]

    if prediction == "strong_sred":
        questions.extend([
            "What specific technological uncertainty was the team trying to resolve?",
            "What known methods, tools, or approaches were attempted first?",
            "Why were the known methods insufficient?",
            "What experiments or iterations were performed?",
            "What were the measurable results of each experiment?",
            "What technical knowledge was gained from the failed or successful attempts?",
            "What evidence exists, such as Jira tickets, Git commits, test logs, benchmarks, or design documents?",
            "Can the client provide the actual supporting records, such as Jira tickets, Git commits, benchmark reports, test logs, design documents, or experiment notes?"
        ])

    elif prediction == "borderline":
        questions.extend([
            "Was there a technological uncertainty, or was this mainly optimization work?",
            "What performance, scalability, accuracy, latency, or reliability target was difficult to achieve?",
            "Were multiple technical approaches tested?",
            "Did any standard methods fail?",
            "What evidence shows the testing process and results?"
        ])

    elif prediction == "needs_more_info":
        questions.extend([
            "What was the actual technical problem being solved?",
            "What was unknown at the start of the work?",
            "Which standard approaches were considered or tested?",
            "What failed, and why?",
            "What experiments, prototypes, or technical investigations were performed?",
            "What new technical knowledge was gained?"
        ])

    elif prediction == "routine":
        questions.extend([
            "Was this work completed using standard tools or known methods?",
            "Were there any failed attempts or unexpected technical limitations?",
            "Was there any uncertainty that could not be resolved through standard engineering practice?",
            "Did the team perform experiments, prototypes, or systematic testing?"
        ])

    if uncertainty_signals:
        questions.append(
            f"The description contains possible uncertainty signals: {', '.join(uncertainty_signals)}. Ask the client to explain these in detail."
        )

    if routine_signals:
        questions.append(
            f"The description also contains routine implementation signals: {', '.join(routine_signals)}. Confirm whether the work went beyond standard implementation."
        )

    return questions

def generate_cra_reference_questions():
    questions = []

    for key, item in CRA_ELIGIBILITY_POINTS.items():
        questions.append(item["question"])

    return questions
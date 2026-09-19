from analysis_engine import BASE_DIR, MODEL_PATH, analyze_text, load_classifier
from case_store import save_intake_case_file
from report_writer import save_report


DEFAULT_MAX_QUESTIONS = 5
STOP_COMMANDS = {"done", "quit", "exit"}


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_section(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def select_intake_questions(agent_assessment, max_questions=DEFAULT_MAX_QUESTIONS):
    questions = []

    for question in agent_assessment["interview_focus"]:
        if question not in questions:
            questions.append(question)

        if len(questions) == max_questions:
            break

    return questions


def build_updated_description(original_text, answers):
    answered_items = [
        item
        for item in answers
        if item["answer"].strip()
    ]

    if not answered_items:
        return original_text.strip()

    lines = [
        original_text.strip(),
        "",
        "Follow-up intake answers:",
    ]

    for index, item in enumerate(answered_items, start=1):
        lines.append(f"{index}. Question: {item['question']}")
        lines.append(f"   Answer: {item['answer'].strip()}")

    return "\n".join(lines)


def save_analysis_report(text, analysis):
    return save_report(
        text,
        analysis["prediction"],
        analysis["probabilities"],
        analysis["labels"],
        analysis["signals"],
        analysis["explanation"],
        analysis["agent_assessment"],
        analysis["recommendation"],
        analysis["evidence_map"],
        analysis["cra_check"],
        analysis["questions"],
        analysis["cra_reference_questions"],
        BASE_DIR,
    )


def print_case_snapshot(title, analysis):
    agent_assessment = analysis["agent_assessment"]

    print_section(title)
    print(f"Prediction: {analysis['prediction']}")
    print(f"Case stage: {agent_assessment['case_stage']}")
    print(f"Priority: {agent_assessment['priority']}")
    print(f"Decision: {agent_assessment['decision']}")
    print(f"Confidence: {agent_assessment['confidence']:.2f}")

    print("\nTop blockers:")
    for blocker in agent_assessment["blockers"][:3]:
        print(f"- {blocker}")

    print("\nNext evidence requests:")
    for item in agent_assessment["evidence_requests"][:5]:
        print(f"- {item}")


def collect_intake_answers(questions):
    answers = []

    for index, question in enumerate(questions, start=1):
        print(f"\nQuestion {index}: {question}")
        print("Answer, press Enter to skip, or type 'done' to finish.")

        try:
            answer = input("> ")
        except EOFError:
            break

        normalized_answer = answer.strip().lower()
        if normalized_answer in STOP_COMMANDS:
            break

        if not answer.strip():
            print("Skipped.")
            continue

        answers.append({
            "question": question,
            "answer": answer.strip(),
        })

    return answers


def run_intake_session(initial_text, model, max_questions=DEFAULT_MAX_QUESTIONS):
    initial_analysis = analyze_text(initial_text, model)
    questions = select_intake_questions(
        initial_analysis["agent_assessment"],
        max_questions=max_questions,
    )

    print_case_snapshot("INITIAL CASE ASSESSMENT", initial_analysis)

    print_section("GUIDED INTAKE QUESTIONS")
    if not questions:
        print("No intake questions were generated.")
        answers = []
    else:
        answers = collect_intake_answers(questions)

    updated_text = build_updated_description(initial_text, answers)
    final_analysis = analyze_text(updated_text, model)
    report_path = save_analysis_report(updated_text, final_analysis)
    session = {
        "initial_analysis": initial_analysis,
        "questions": questions,
        "answers": answers,
        "updated_text": updated_text,
        "final_analysis": final_analysis,
        "report_path": report_path,
    }
    case_file_path = save_intake_case_file(session, BASE_DIR)
    session["case_file_path"] = case_file_path

    print_case_snapshot("UPDATED CASE ASSESSMENT", final_analysis)

    print_section("UPDATED CASE DESCRIPTION")
    print(updated_text)

    print_section("REPORT SAVED")
    print(report_path)

    print_section("CASE FILE SAVED")
    print(case_file_path)

    return session


def main():
    model = load_classifier(MODEL_PATH)

    print("\nSR&ED Guided Intake Agent")
    print("Paste an initial project description.")
    print("The agent will ask follow-up questions and re-run the assessment.")
    print("Type 'quit' as the project description to stop.")

    while True:
        try:
            initial_text = input("\nProject description: ")
        except EOFError:
            print("\nGoodbye.")
            break

        if initial_text.strip().lower() in STOP_COMMANDS:
            print("\nGoodbye.")
            break

        if not initial_text.strip():
            print("Please enter some text.")
            continue

        run_intake_session(initial_text, model)


if __name__ == "__main__":
    main()

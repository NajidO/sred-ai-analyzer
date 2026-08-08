def generate_recommendation(prediction, probabilities, labels, signals):
    probability_map = dict(zip(labels, probabilities))

    strong_score = probability_map.get("strong_sred", 0)
    borderline_score = probability_map.get("borderline", 0)
    routine_score = probability_map.get("routine", 0)
    needs_more_info_score = probability_map.get("needs_more_info", 0)

    uncertainty_signals = signals["uncertainty_signals"]
    routine_signals = signals["routine_signals"]

    if prediction == "strong_sred":
        return (
            "This looks like a strong SR&ED candidate, but it still needs evidence. "
            "Focus on documenting the technological uncertainty, failed standard approaches, experiments, results, and technical learning."
        )

    if prediction == "borderline":
        return (
            "This looks borderline. It may be SR&ED if the work involved technological uncertainty and systematic experimentation. "
            "Ask follow-up questions to separate routine optimization from experimental development."
        )

    if prediction == "needs_more_info":
        return (
            "There is not enough detail to assess SR&ED potential. "
            "The next step is to ask what was technically unknown, what approaches failed, and what experiments were performed."
        )

    if prediction == "routine":
        if uncertainty_signals:
            return (
                "The model predicts routine work, but some uncertainty signals were detected. "
                "Review carefully to see whether the work went beyond standard implementation."
            )

        return (
            "This looks mostly routine based on the current description. "
            "It may not support SR&ED unless there were unresolved technological uncertainties and experimental work not yet described."
        )

    return (
        "Review manually. The model result is unclear, and more technical detail may be needed."
    )
# SR&ED Technical Uncertainty Analyzer

Python prototype for analyzing technical project descriptions for possible Canadian SR&ED indicators.

This tool does not make eligibility decisions. It supports analyst review by classifying a description, surfacing technical uncertainty signals, identifying routine implementation signals, suggesting follow-up questions, and mapping evidence gaps.

## Purpose

The analyzer is designed around common SR&ED review concepts:

- whether the work appears routine or technically uncertain
- whether the description suggests technological advancement
- whether there was systematic investigation through testing or analysis
- whether the project records show results, learning, and supporting evidence
- whether more information is needed before an analyst can assess the claim

## Labels

The classifier predicts one of four labels:

- `routine`: standard implementation, configuration, migration, reporting, integration, or use of known tools
- `borderline`: some technical work is described, but uncertainty, experiments, learning, or evidence are incomplete
- `needs_more_info`: the description is too vague to assess
- `strong_sred`: the description includes failed standard methods, technological uncertainty, testing or analysis, and technical learning

## Project Structure

```text
data/
  sred_training_data.csv
  test_examples.csv
src/
  add_cra_batch.py
  train_sred_classifier.py
  run_tests.py
  predict_sred.py
  cra_guideline_checker.py
  cra_reference.py
  evidence_mapper.py
  questions.py
  recommendation.py
  report_writer.py
  rules.py
requirements.txt
```

Generated reports, virtual environments, Python caches, and local trained model artifacts are excluded from GitHub by `.gitignore`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the Classifier

```bash
python src/train_sred_classifier.py
```

This trains the sklearn pipeline from `data/sred_training_data.csv` and writes `sred_classifier.joblib`.

## Run Tests

```bash
python src/run_tests.py
```

The test runner first validates that `data/sred_training_data.csv` can be loaded safely with pandas and has the expected `text,label` columns. It then runs the classifier against `data/test_examples.csv`.

## Run the Analyzer

```bash
python src/predict_sred.py
```

Paste a technical project description when prompted. The tool returns a classification, probabilities, signals, follow-up questions, and a saved report.

## Add CRA-Grounded Training Examples

```bash
python src/add_cra_batch.py
```

The batch script uses pandas DataFrames instead of raw CSV appends, skips duplicate examples by `text`, writes with `to_csv(index=False)`, and prints row and label counts before and after.

## Current Local Validation

Latest local run:

```text
Training CSV integrity check: PASS
Training rows: 147
Classifier tests: 18/18
Accuracy: 100.00%
```

This accuracy is only for the current controlled test set. More diverse examples are still needed before treating the model as reliable.

## Next Improvements

- add more challenging `borderline` and `needs_more_info` tests
- separate data validation into a reusable module or test file
- add analyst-facing examples and confidence interpretation
- consider a small Streamlit interface for guided review
- document model limitations and human review requirements

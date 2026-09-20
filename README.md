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
  agent_assessment.py
  analysis_engine.py
  case_store.py
  case_manager.py
  technical_report.py
  train_sred_classifier.py
  run_tests.py
  intake_agent.py
  predict_sred.py
  cra_guideline_checker.py
  cra_reference.py
  evidence_mapper.py
  explanation.py
  questions.py
  recommendation.py
  report_writer.py
  rules.py
requirements.txt
```

Generated reports, guided intake case files, virtual environments, Python caches, and local trained model artifacts are excluded from GitHub by `.gitignore`.

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

## Run the Guided Intake Agent

```bash
python src/intake_agent.py
```

Paste an initial project description. The agent runs an initial assessment, asks the highest-priority follow-up questions, adds the answers to the case description, re-runs the assessment, saves an updated report, and writes a reusable JSON case file under `case_files/`.

## Manage Saved Case Files

```bash
python src/case_manager.py list
python src/case_manager.py inspect case_YYYYMMDD_HHMMSS
python src/case_manager.py resume case_YYYYMMDD_HHMMSS
```

Saved case files can be listed, inspected, and resumed. Resuming a case asks follow-up questions from the latest assessment, saves a new report, and writes a child JSON case file linked to the parent case.

## Generate A T661 Technical Report Draft

```bash
python src/case_manager.py report case_YYYYMMDD_HHMMSS
python src/technical_report.py case_YYYYMMDD_HHMMSS --show
```

The technical report generator turns a saved case file into a Markdown SR&ED technical report draft under `technical_reports/`. The main output is structured around Form T661 Part 2, Section B:

- line 242: scientific or technological uncertainties, maximum 350 words
- line 244: work performed in the tax year, maximum 700 words
- line 246: scientific or technological advancements, maximum 350 words

The generator also includes word counts, gap warnings, report readiness, and supporting analyst notes. Questionnaire-style inputs are parsed into the T661 lines before the broader reviewer notes are shown.

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
Classifier tests: 26/26
Accuracy: 100.00%
Agentic assessment checks: PASS
Guided intake agent checks: PASS
Reusable JSON case file checks: PASS
Case manager checks: PASS
Technical report generator checks: PASS
T661 project description checks: PASS
```

This accuracy is only for the current controlled test set. More diverse examples are still needed before treating the model as reliable.

## Next Improvements

- add more challenging `borderline` and `needs_more_info` tests
- separate data validation into a reusable module or test file
- add analyst-facing examples and confidence interpretation
- add case export formats for analyst handoff
- improve T661 line 242/244/246 wording through real case testing
- consider a small Streamlit interface for guided review
- document model limitations and human review requirements

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
benchmarks/
  BENCHMARK_FINDINGS.md
  semantic_evidence_gate_benchmark.json
  t661_adversarial_benchmark.json
  t661_capability_benchmark.json
  t661_holdout_benchmark.json
src/
  add_cra_batch.py
  agent_assessment.py
  analysis_engine.py
  case_store.py
  case_manager.py
  evidence_agent.py
  grounding.py
  responses_http_client.py
  technical_report.py
  train_sred_classifier.py
  run_tests.py
  intake_agent.py
  llm_report_agent.py
  run_capability_benchmark.py
  run_semantic_evidence_benchmark.py
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

The technical report generator first assesses whether a saved case contains enough
grounded information for Form T661 Part 2, Section B. For each line it lists supported
source facts, missing information, and TU/SIS-specific client questions. It only
creates report prose when all three lines are sufficiently supported:

- line 242: scientific or technological uncertainties, maximum 350 words
- line 244: work performed in the tax year, maximum 700 words
- line 246: scientific or technological advancements, maximum 350 words

When evidence is incomplete, the generator writes a T661 evidence assessment and
explicitly withholds the report draft. It asks about the exact unknown, known methods,
alternatives considered and tested, controlled conditions, measurements, failures,
abandoned paths, pivots, further experiments, technical learning, and unresolved
year-end uncertainty. Complete cases receive word-count checks, gap warnings, report
readiness, and supporting analyst notes.

Before drafting the T661 lines, the agent adds a drafting strategy and rationale. It decides whether the source facts are better handled as one integrated narrative or split into TU/SIS streams, identifies candidate technical uncertainty streams, and asks more specific follow-up questions tied to detected project facts.

When the agent decides that splitting is clearer, the T661 draft itself is sectioned with labels such as `TU1`, `TU2`, `SIS1`, and `SIS2` inside lines 242, 244, and 246 so the submitted technical narrative can separate overlapping uncertainties and investigations.

## Run The Evidence-First Report Agent

The optional OpenAI path is the main semantic report agent. It combines the local
classifier and deterministic safeguards with four separate model stages when the
evidence is complete:

1. Evidence extraction creates typed technical streams and exact source-quote-backed
   evidence items. Each item records certainty, claimed/prior/future timing, and whether
   the work belongs to the claimant, a claimant-directed contractor, or a third party.
   Local code derives line, column, and character locations and rejects repeated quotes
   whose occurrence cannot be identified unambiguously. It also extracts explicit SIS
   chains linking uncertainty, hypothesis, performed work, result, conclusion, and
   advancement.
2. An independent evidence-audit call checks the claimed tax year, every extracted
   item's normalized fact, category, certainty, tax-year scope, attribution, and stream
   assignment, every SIS relationship and chronology, and every extracted routine-work
   or attribution blocker. Unsupported or ambiguous years, items, sequences, or blockers
   are removed, and newly detected contradictions block affected lines.
3. A local readiness gate checks every stream against the evidence needed for Lines
   242, 244, and 246. It requires a uniquely grounded four-digit claimed tax year and a
   coherent audited SIS chain for Lines 244 and 246. It blocks contradictory, future,
   unattributed, routine, disconnected, or incomplete evidence and generates targeted
   questions instead of a partial report.
4. A deterministic structure planner selects integrated, split-stream, or hybrid
   presentation from the validated TU/SIS topology. Multi-stream and multi-sequence
   plans define required line-specific TU/SIS labels and ordering.
5. Grounded drafting runs only after every line passes. The model must follow the
   structure plan and cite evidence
   IDs for each line and for every sentence or standalone factual statement.
6. A local post-draft audit rejects the wrong structure mode, missing or reordered
   TU/SIS sections, empty headings, claims placed under the wrong TU/SIS label,
   cross-sequence evidence mixing, missing sentence maps, unknown or out-of-scope
   evidence IDs, missing category support, over-limit prose, and numeric facts not
   present in the client source.
7. An independent grounding call checks every drafted claim against exactly its cited
   evidence. Any ambiguous, unsupported, omitted, or incompletely reviewed claim causes
   all T661 prose to be withheld.

The command runs from your Mac and saves an editable Markdown report containing the
evidence ledger, readiness decision, TU/SIS structure rationale, questions or draft,
the support IDs used for each drafted line and claim, and the independent audit result.
No website or hosting is required.

Install the updated requirements and pass a UTF-8 text or Markdown project description:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python src/llm_report_agent.py /path/to/project.txt --show
```

The agent uses the official OpenAI Responses API. When the optional `openai` Python
package is installed it uses that SDK; otherwise it automatically uses the included
standard-library HTTPS client, so no additional package is required for model calls.
Both transports send the same strict Structured Outputs request. See the official
[Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

If `OPENAI_API_KEY` is not already set, the command asks for the key securely. Terminal
does not display characters while you paste or type the key; press Return when finished.
The key is used for that process only and is not saved. Do not place it in source code
or commit it to GitHub.

You can alternatively set the key for the current shell session first:

```bash
export OPENAI_API_KEY="your-api-key"
```

The default model is `gpt-5.6-terra`. Override it for a comparison without changing code:

```bash
OPENAI_MODEL="gpt-5.6-sol" venv/bin/python src/llm_report_agent.py /path/to/project.txt --show
```

Reasoning effort defaults to `high` and can be changed for a cost/quality comparison:

```bash
venv/bin/python src/llm_report_agent.py /path/to/project.txt \
  --reasoning-effort medium --show
```

To confirm that the local classifier and report preparation work without making a paid
API request:

```bash
venv/bin/python src/llm_report_agent.py /path/to/project.txt --dry-run
```

The model calls use strict structured outputs. Source text is treated as untrusted
client evidence, so instructions embedded inside it do not override the agent. Exact
quote validation and numeric grounding are deterministic local checks. Independent
audits check both evidence classification and final claim support. These controls reduce
unsupported drafting; they cannot verify whether the client's statement or record is
true. Human technical and tax review is mandatory.

## Test An Incomplete Client Intake

The repository includes a fictional SEM project draft with ten evidence categories
left incomplete on purpose. Run the local evaluator to see which omissions the agent
detects and which specific follow-up questions it asks:

```bash
venv/bin/python src/capability_eval.py \
  examples/incomplete_sem_client_draft.md \
  examples/incomplete_sem_expected_gaps.json \
  --output reports/incomplete_sem_evidence_assessment.txt
```

The expected-gaps JSON is used only to score the result after analysis; it is not
included in the project information supplied to the analyzer. This is an offline
test of the local classifier, report strategy, and evidence checks. This fixture must
produce `needs_more_information` and no T661 draft. It does not make an OpenAI API
request.

## Run The T661 Capability Benchmarks

The repository includes three offline benchmark sets covering complete, incomplete,
routine, contradictory, negated, qualitative, multi-stream, unstructured,
prompt-injection, failed-project, future-work, attribution, tax-year, recollection,
and analysis-without-physical-prototype cases:

```bash
venv/bin/python src/run_capability_benchmark.py
venv/bin/python src/run_capability_benchmark.py \
  benchmarks/t661_holdout_benchmark.json
venv/bin/python src/run_capability_benchmark.py \
  benchmarks/t661_adversarial_benchmark.json
```

The normal test runner executes all three sets as regression checks. The cases test the
local sklearn classifier, evidence gate, stream planner, and deterministic drafting
controls. They do not call an OpenAI model and do not establish legal eligibility.
See `benchmarks/BENCHMARK_FINDINGS.md` for the baseline failures, fixes, current
results, and limits of the evaluation.

The semantic evidence layer has a separate 16-case benchmark for claimed-year grounding,
SIS linkage, chronology, attribution, future work, contradictions, and evidence-driven
integrated, split-stream, and hybrid structure selection:

```bash
venv/bin/python src/run_semantic_evidence_benchmark.py
```

This benchmark exercises the deterministic validation and independent-audit application
layer with controlled evidence graphs. It does not score a live model's extraction
quality.

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
Report strategy planner checks: PASS
Semantic evidence-agent checks: PASS
T661 capability benchmark: 29/29
Semantic evidence and structure benchmark: 16/16
Incomplete intake gaps detected: 10/10
Incomplete intake T661 drafts generated: 0
Complete questionnaire T661 drafting: PASS
Unstructured intake assessment: PASS
```

The classifier accuracy and benchmark scores are only for small, synthetic controlled
sets. They are regression signals, not estimates of production accuracy. Real,
independently labelled project records and analyst review are still required before
treating the analyzer as reliable.

## Known Limits And Next Validation

- The semantic path requires an OpenAI API key. It makes extraction and independent
  evidence-audit calls for every case; when the evidence gate passes it also makes a
  drafting call and an independent claim-grounding call. This increases cost and latency
  in exchange for stricter evidence- and claim-level support.
- Offline benchmarks are synthetic regression tests, not production accuracy or an
  eligibility opinion.
- Exact quotes prove only that a statement appeared in the supplied source. They do
  not authenticate records, dates, measurements, authorship, or client claims.
- The next meaningful evaluation is a blinded set of de-identified real projects,
  independently reviewed by experienced SR&ED practitioners. Measure false-ready
  rate, false-block rate, unsupported-fact rate, stream quality, question usefulness,
  and reviewer agreement before packaging this as a client-facing product.
- consider a small Streamlit interface for guided review
- document model limitations and human review requirements

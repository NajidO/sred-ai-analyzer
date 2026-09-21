# T661 Capability Benchmark Findings

## Purpose

These benchmarks test whether the local analyzer can decide when source information
is sufficient to draft Form T661 lines 242, 244, and 246, when it should ask for more
information, and when apparently technical work is routine or internally
contradictory.

The criteria were checked against the CRA's current T4088 guide and project-description
guidance:

- Line 242: identify the scientific or technological uncertainty, the objective or
  capability sought, the starting knowledge base, and why generally available
  knowledge could not resolve the uncertainty.
- Line 244: describe the claimed-year work chronologically, including hypotheses,
  experiments or analysis, results, and conclusions.
- Line 246: describe the new scientific knowledge or technological advancement
  achieved or attempted, rather than a business benefit.
- Apply the published maximums of 350, 700, and 350 words.

Official references:

- https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/t4088/guide-form-t661-scientific-research-experimental-development-expenditures-claim-guide-form-t661.html
- https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program/sred-claim/group-projects/project-description-t661.html

The analyzer also uses conservative product safeguards that go beyond the wording of
the three description fields. It blocks automatic drafting when supporting records
are explicitly unavailable, when the source contains material contradictions, or when
claimed-year chronology is denied. Those are anti-hallucination and review controls,
not statements that the CRA form itself requires a particular document type in each
description field.

## Evaluation Design

The tuning set contains 11 synthetic projects across software and non-software
domains. It includes complete records, omitted results, missing advancement,
qualitative outcomes, routine vendor configuration, contradictions, negated records,
format variation, multiple uncertainty streams, and an instruction embedded in
client text that attempts to bypass the evidence gate.

The seven-case holdout was written before the generalization fixes and run once
without tuning. It tested unseen robotic, ceramic, fluid-control, data-migration, and
mixed-domain wording, plus fiscal-year contradictions and unavailable evidence.

Each case asserts the expected draft decision, blocked and ready T661 lines, minimum
stream count, optional classifier label range, and whether a draft invents a
measurement not found in the source.

## Results

| Evaluation | Before fixes | Current |
| --- | ---: | ---: |
| Tuning benchmark | 4/11 (36%) | 11/11 (100%) |
| Originally untouched holdout | 4/7 (57%) | 7/7 (100%) |
| Combined current regression set | 8/18 (44%) | 18/18 (100%) |
| Classifier examples | Not changed | 26/26 (100%) |

The current scores show that the named failures are covered by regression tests. They
do not show 100% real-world accuracy because all 18 capability cases are synthetic,
the post-fix holdout is no longer independent, and the classifier test set has only 26
examples.

## What Failed And Why

1. Complete evidence outside the original domains was missed. Stream planning relied
   on four hard-coded keyword groups and could produce only one generic fallback.
2. Valid Line 244 facts were missed when alternatives were written as counted designs
   (for example, five filters) rather than words such as "multiple" or "variants."
3. Qualitative experimental observations were treated as incomplete when no numeric
   unit appeared.
4. A named record could count as support even when the same sentence said the files
   were unavailable or could not be produced. Negation scope was too narrow.
5. Routine vendor configuration could look experimental because it contained tests,
   alternatives, and metrics even though documented settings resolved the objective.
6. Explicit claimed-year contradictions were not recognized unless they used one of a
   few exact phrases.
7. A statement about uncertainty remaining at year-end could be misclassified as a
   separate technical uncertainty stream.
8. Percent measurements were missed because the measurement regular expression used
   a word boundary after the percent sign.

## Changes Made

- Added source-derived uncertainty clauses so unfamiliar and mixed domains can produce
  grounded TU streams without a domain template.
- Added evidence checks for enumerated alternatives, qualitative outcomes, failure and
  pivot language, fiscal-year denials, and scoped record negation.
- Prevented remaining year-end uncertainty from becoming a duplicate TU stream.
- Added a narrow routine-work guardrail for documented vendor or standard
  configuration that resolved the stated objective.
- Fixed percent and measured-outcome recognition.
- Added machine-readable benchmark results and made both benchmark files part of the
  normal test suite.

## Remaining Limits

- The local path is a sklearn classifier plus deterministic rules. It does not reason
  like the optional OpenAI report agent, and this benchmark did not make an API call.
- The checks infer evidence from language. They cannot verify that a test, measurement,
  date, record, or technical conclusion is true.
- Contradiction and negation handling remains pattern-based and can miss novel
  paraphrases, cross-document conflicts, tables, attachments, and implied chronology.
- Stream extraction is lexical. Closely related uncertainties may be split, and
  distinct but similarly worded uncertainties may be merged.
- The test corpus has no independent CRA decisions or adjudicated real claims. It
  cannot measure precision, recall, or acceptance likelihood in production.
- The suite does not yet cover multilingual intake, OCR errors, very long evidence
  packages, multi-year continuations, contractor attribution, or conflicts across
  multiple uploaded documents.
- Human technical and tax review remains mandatory. The tool should organize evidence,
  block unsupported drafting, and expose questions; it should not make the final
  eligibility decision.

## Next Benchmark Step

Build a blinded set from de-identified historical project records labelled independently
by at least two experienced SR&ED reviewers. Measure line-level false-ready and
false-block rates, stream quality, unsupported-fact rate, question usefulness, and
reviewer agreement. Keep that set outside development and run it only at release
candidates so it remains a genuine generalization test.

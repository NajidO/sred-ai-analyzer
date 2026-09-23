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

The seven-case holdout was written before the first generalization fixes and run once
without tuning. It tested unseen robotic, ceramic, fluid-control, data-migration, and
mixed-domain wording, plus fiscal-year contradictions and unavailable evidence.

An additional 11-case adversarial set targets criteria that simple keyword gates often
get wrong: attempted advancement in an unsuccessful project, one valid systematic
analysis path, a precise qualitative objective, no remaining year-end uncertainty,
future work presented as completed, a commercial A/B test, implementation-only work
in the claimed year, an unverified remembered result, an implicit two-stream project,
vendor work attributed to the claimant, and simulation analysis without a physical
prototype. Its baseline was recorded before the fixes described below.

Each case asserts the expected draft decision, blocked and ready T661 lines, minimum
and sometimes maximum stream count, prohibited stream titles, optional classifier
label range, and whether a draft invents a measurement not found in the source.

## Results

| Evaluation | Before fixes | Current |
| --- | ---: | ---: |
| Tuning benchmark | 4/11 (36%) | 11/11 (100%) |
| Originally untouched holdout | 4/7 (57%) | 7/7 (100%) |
| Adversarial evidence benchmark | 4/11 (36%) | 11/11 (100%) |
| Combined current regression set | 12/29 (41%) | 29/29 (100%) |
| Classifier examples | Not changed | 26/26 (100%) |

The current scores show that the named failures are covered by regression tests. They
do not show 100% real-world accuracy because all 29 capability cases are synthetic,
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
9. Alternatives, a failed path, a quantified objective, and unresolved year-end work
   were treated as mandatory even though a source can support the T661 lines without
   every one of those narrative details.
10. Future plans and earlier-year investigations could satisfy the work gate when the
    prose contained experimental vocabulary.
11. Vendor experiments and unverified recollections could be mistaken for claimant
    evidence.
12. Valid mathematical or simulation analysis could be blocked because no physical
    test was described.
13. Broad domain keywords could create extra technical streams unrelated to the
    source-defined uncertainty.

## Changes Made

- Added source-derived uncertainty clauses so unfamiliar and mixed domains can produce
  grounded TU streams without a domain template.
- Added evidence checks for enumerated alternatives, qualitative outcomes, failure and
  pivot language, fiscal-year denials, and scoped record negation.
- Prevented remaining year-end uncertainty from becoming a duplicate TU stream.
- Added a narrow routine-work guardrail for documented vendor or standard
  configuration that resolved the stated objective.
- Fixed percent and measured-outcome recognition.
- Separated blocking evidence requirements from useful but advisory detail. A precise
  qualitative objective, one systematic hypothesis path, advancement attempted through
  failed work, and a completed project with no remaining uncertainty can now pass when
  their required evidence is present.
- Excluded future plans from completed work, added claimed-year implementation and
  recollection conflicts, and recognized systematic analysis without requiring a
  physical prototype.
- Tightened domain stream anchors, added semantic Markdown heading parsing, and split
  implicit linked uncertainties expressed as separate relationships.
- Added maximum-stream and prohibited-title assertions so a passing benchmark cannot
  hide unrelated extra streams.
- Added the adversarial benchmark and made all three benchmark files part of the normal
  test suite.

## Semantic Agent Safeguards

The optional OpenAI path now uses a strict evidence-extraction schema followed by an
independent evidence-classification audit before readiness. Every extracted item must
contain an exact source quote and records its certainty, tax-year scope, technical
stream, category, and attribution. Local code rejects fabricated quotes, unsupported
normalized numbers, stream-specific evidence assigned globally, and obvious
claimed-year conflicts. The independent audit reviews every item across six semantic
dimensions, removes unsupported or ambiguous classifications, and can add material
contradictions that the extraction stage missed.

The local readiness gate requires objective, starting knowledge, standard-practice
limit, and uncertainty evidence for Line 242; hypothesis, claimed-year investigation,
result, conclusion, and supporting-record evidence for Line 244; and claimed-year
advancement evidence for Line 246. Drafting occurs only after all streams pass the
evidence audit and readiness gate. A local post-draft audit requires evidence IDs for
every category,
stream, and sentence; checks word limits and numeric grounding; and rejects incomplete
or out-of-scope claim maps. A final independent model call reviews every sentence
against exactly its cited evidence. Any ambiguous, unsupported, missing, duplicated, or
partially reviewed claim causes all T661 prose to be withheld. Mocked end-to-end tests
exercise these stages without making a paid API call.

## Remaining Limits

- The benchmark score covers the sklearn and deterministic local path. Semantic-agent
  tests use controlled mock model responses and do not establish real-model accuracy.
- The checks infer evidence from language. They cannot verify that a test, measurement,
  date, record, or technical conclusion is true.
- Contradiction and negation handling remains pattern-based and can miss novel
  paraphrases, cross-document conflicts, tables, attachments, and implied chronology.
- Offline stream extraction remains partly lexical. The OpenAI path can organize
  unfamiliar domains semantically, but model-created streams still require exact
  supporting evidence and can be over-split or merged.
- The test corpus has no independent CRA decisions or adjudicated real claims. It
  cannot measure precision, recall, or acceptance likelihood in production.
- The suite does not yet cover multilingual intake, OCR errors, very long evidence
  packages, multi-year continuations across several files, or conflicts across
  multiple uploaded documents. Contractor and third-party attribution are covered at
  a unit level but still need real-case validation.
- Human technical and tax review remains mandatory. The tool should organize evidence,
  block unsupported drafting, and expose questions; it should not make the final
  eligibility decision.

## Next Benchmark Step

Build a blinded set from de-identified historical project records labelled independently
by at least two experienced SR&ED reviewers. Measure line-level false-ready and
false-block rates, stream quality, unsupported-fact rate, question usefulness, and
reviewer agreement. Keep that set outside development and run it only at release
candidates so it remains a genuine generalization test.

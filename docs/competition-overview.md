# EXACT 2026 Competition Overview

Source checked on June 20, 2026:

- Main page: https://exact-ijcnn.vercel.app/exact
- IEEE competition listing linked from the main page: https://attend.ieee.org/wcci-2026/competitions/

## Challenge scope

EXACT 2026 is the second International XAI Challenge for Transparent Educational Question-Answering. The task is to build a QA system that returns:

- a correct final answer
- a non-empty natural-language explanation
- optional structured reasoning evidence

The competition explicitly accepts symbolic, neuro-symbolic, fine-tuned LLM, or hybrid systems. Symbolic reasoning is encouraged but not mandatory.

## Hard rules

- All LLMs must be open-source.
- Each LLM must have 8B parameters or fewer.
- Closed-source models such as GPT, Claude, and Gemini are prohibited.
- External datasets used for fine-tuning or symbolic components must be disclosed.

## Dataset types

### Type 1: logic-based educational queries

- 464 records
- 915 total questions
- domain: academic regulations, grading, enrollment, scholarship, requirements
- input at evaluation time:
  - `question`
  - natural-language premises
  - optional answer choices

Training data may also contain FOL, explanations, and other annotations, but those are reference annotations rather than required runtime inputs.

### Type 2: physics problems

- 1,755 text-only problems
- domain: electric circuits and electrostatics
- typical topics: resistance, voltage, current, power, capacitance, electric field, energy
- input at evaluation time:
  - question only

Type 2 answers are numerical or short textual outputs with units where applicable.

## Evaluation dimensions

The official page defines three dimensions:

- `P1`: correctness of answers
- `P2`: supporting premises for Type 1
- `P3`: depth of reasoning

Current hidden-test round details from the site:

- hidden test set contains 50 Type 1 queries and 50 Type 2 queries
- Type 1 automatic scoring uses:
  - answer correctness
  - supporting-premise correctness
- Type 2 automatic scoring uses:
  - numerical answer
  - unit
- explanations are required for schema completeness, but are not part of the automated hidden-test score
- in the final live round, chairs assess answer quality, explanations, and reasoning depth in real time

## Required API response shape

The official page requires the API to return a JSON list even for a single query.

Required fields per item:

- `query_id`
- `answer`
- `unit`
- `explanation`
- `premises_used`
- `reasoning`

Important notes from the official page:

- `explanation` must be non-empty
- for Type 1, `premises_used` is scored against gold supporting-premise indices

## Submission package

Each team must submit:

- an API endpoint
- a one-page solution description

The one-page description should explain:

- approach
- models used
- dataset used for training

## Timeline

Dates checked on June 20, 2026:

- Main competition phase: May 5 to June 14, 2026
- Phase 1 evaluation: June 14 to June 15, 2026
- Model refinement period: June 16 to June 17, 2026
- Phase 2 evaluation and ranking announcement: June 18 to June 20, 2026
- Public test day, solution presentations, and final result release: June 25, 2026
- Paper submission for top 10 teams: June 30 to July 15, 2026
- On-site presentation at CSoNet 2026: November 16 to November 18, 2026

## Practical implications for this repo

- Type 1 should prioritize exact answer strings and accurate `premises_used`.
- Type 2 should prioritize numeric correctness and unit normalization.
- Non-empty explanations remain mandatory even when they are not auto-scored in the hidden round.
- Rich structured reasoning still matters for the public/final live evaluation.

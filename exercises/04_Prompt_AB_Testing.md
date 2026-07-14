# Exercise 4 — Prompt Versioning and A/B Testing

**Difficulty:** Advanced | **Time:** 15–20 min | **Theory link:** 3.10 Prompt Evaluation, Versioning, and Reusable Prompt Libraries

---

## 🎯 Goal

Two prompts can produce noticeably different quality on the same task, and "which one is better" shouldn't be decided by gut feeling. This exercise writes two competing prompt variants, scores both against a rubric using an LLM judge, and picks a winner with evidence — the same discipline behind tools like Promptfoo, built here in plain Python first so the underlying idea isn't a black box.

By the end, you will be able to:
- Write two meaningfully different prompt variants for the same task
- Score generated outputs against a defined rubric using an LLM-as-judge pattern
- Aggregate scores across a test set to make an evidence-based "which prompt wins" decision
- Explain how this manual process maps onto dedicated prompt-testing tools used in production (Module 6, Promptfoo)

**A note on tooling:** production teams typically use a dedicated framework — **Promptfoo** is the one named in this course's software list — to automate exactly this kind of comparison at scale, with a proper test-runner, CI/CD integration, and richer reporting than we build by hand here. This exercise deliberately builds the underlying logic yourself first, in plain Python, so Promptfoo (introduced properly in Module 6) reads as "the production version of something I already understand," not a mysterious new black box.

---

## 🧠 Concept Primer

### Why does prompt PHRASING alone change output quality this much?

The two variants in this exercise ask for the exact same underlying task (turn a policy document into an FAQ answer) but differ in specificity: Variant A gives almost no guidance, Variant B specifies a role, explicit length constraint, a jargon rule, and a requirement to end with an actionable takeaway. More explicit constraints generally produce more consistent, usable output — vague prompts leave more decisions up to the model's default behaviour, which varies.

### What is "LLM-as-judge," and why use it here instead of a human?

Having a human manually score every output against every rubric dimension doesn't scale — even in this tiny 5-document exercise, that's 10 outputs × 3 dimensions = 30 judgments. **LLM-as-judge** uses a second LLM call, given the rubric explicitly, to score each output — trading some judgment accuracy for the ability to scale to hundreds or thousands of test cases. It's not perfect (LLM judges can have their own biases, covered properly in Module 6), but it's dramatically more scalable than manual review, and good enough for iterating on a prompt during development.

### Why generate AND judge with separate prompts/calls?

Keeping generation and judging as two distinct steps (rather than asking one call to "write and then critique itself") keeps the judge's evaluation independent of the specific wording quirks of whichever variant generated the answer — the judge only ever sees the rubric and the final output, not which prompt produced it.

---

## Step 1 — The dataset

`data/policy_documents.csv` — 5 real-style banking policy paragraphs, the "source of truth" both prompt variants are asked to summarise into a customer-facing FAQ answer.

---

## Step 2 — Azure OpenAI setup

Needs one deployment (`AZURE_OPENAI_DEPLOYMENT`). See the package `README.md`.

---

## Step 3 — Code walkthrough

### Stage 1 — Define the two competing prompt variants

```python
PROMPT_VARIANT_A = """Summarise this bank policy document for a customer-facing FAQ page:
{document}
"""

PROMPT_VARIANT_B = """You are a customer communications specialist...
Rules:
- Maximum 2 sentences.
- No internal jargon...
- End with a concrete, actionable takeaway...
{document}
Write the FAQ answer now.
"""
```

**What this does:** Variant A is intentionally minimal — a realistic "first draft" prompt. Variant B adds a role, explicit constraints, and an output requirement — this is the actual A/B test: does specificity measurably improve the result, or is it just extra words that don't change much?

### Stage 2 — Define the judge's rubric

```python
JUDGE_PROMPT_TEMPLATE = """Score the following FAQ answer against its source policy document on three dimensions,
each from 1 (poor) to 5 (excellent). Return ONLY a JSON object with these exact keys:
{{"clarity": <1-5>, "completeness": <1-5>, "conciseness": <1-5>}}
...
"""
```

**What this does:** the SAME rubric is applied to both variants' outputs, by the same judge call structure — consistent scoring criteria is what makes the final comparison meaningful rather than arbitrary. Note the `temperature=0` on the judge call specifically (in the code) — you want the *scoring* to be as consistent as possible, even if the *generation* step uses a bit of temperature for natural variation.

### Stage 3 — Generate, judge, and aggregate

```python
results_df["total_score"] = results_df[["clarity", "completeness", "conciseness"]].sum(axis=1)
...
results_df.groupby("variant")[["clarity", "completeness", "conciseness", "total_score"]].mean().round(2)
```

**What this does:** same `groupby().agg()`-style aggregation pattern you've now seen in Module 2 Exercise 3 — collapsing individual per-document scores down into one summary row per variant, which is the actual basis for declaring a winner.

---

## Expected Output (illustrative)

```
DOC002: In the event of a disputed transaction, customers have 60 days...

-- Variant A --
Disputes must be filed within 60 days of the statement date. Exceptions may be granted at the discretion of the fraud investigation team for extenuating circumstances.
Scores: clarity=3 completeness=4 conciseness=4

-- Variant B --
You have 60 days from your statement date to dispute a transaction — file as soon as you notice it to be safe. Contact us right away if you're past that window, as exceptions are sometimes possible.
Scores: clarity=5 completeness=4 conciseness=5

AVERAGE SCORES BY VARIANT
          clarity  completeness  conciseness  total_score
variant
A            3.2           3.8          3.6         10.6
B            4.6           4.0          4.4         13.0
```
*(Illustrative — exact scores vary by model and run, but Variant B's explicit constraints should generally score at least as well, often better, particularly on clarity.)*

---

## 🛠 Common Pitfalls

- **Judge bias toward longer or shorter answers:** LLM judges can have systematic preferences (e.g., favouring more detailed answers even when the rubric says "conciseness") — a real limitation covered properly in Module 6's "LLM-as-judge" section on calibration.
- **Only running the test set once:** a single run can be noisy — for a real decision, you'd want to run this multiple times and look at the average, not just one pass.
- **Forgetting the judge is ALSO an LLM call that costs money/latency:** at scale, evaluation itself becomes a real line-item cost — worth mentioning if the room asks "why not just judge everything all the time."

---

## 🏠 Homework Exercise

1. Write a **Variant C** — try a completely different strategy (e.g., ask for the answer as a bullet list instead of prose, or ask the model to write it "as if explaining to a friend").
2. Add it to the script's generation/judging loop and re-run against all 5 documents.
3. Write 2–3 sentences: did your new variant win outright, or did it trade off one dimension (e.g., higher clarity) for another (e.g., lower conciseness)? A real prompt-selection decision often isn't a clean sweep — practice articulating the trade-off explicitly rather than declaring a simple "winner."

**Hints:**
- To add Variant C cleanly, follow the exact same pattern as A and B: define the template, call `generate_answer()` and `judge_answer()` with it, append results with `"variant": "C"`.
- If you want a preview of where this goes in production: look up "Promptfoo" after this exercise — you'll recognise every concept (test set, rubric, comparison) from what you just built by hand.

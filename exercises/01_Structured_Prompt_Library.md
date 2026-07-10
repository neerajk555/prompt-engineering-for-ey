# Exercise 1 — Build a Structured Prompt Library for a Business Use Case

**Difficulty:** Beginner–Intermediate | **Time:** 15–20 min | **Theory link:** 3.1 Prompt Structure / 3.4 Structured Output Prompting

---

## 🎯 Goal

Real applications don't send an LLM a vague question and hope for the best — they send a carefully structured prompt with a defined role, explicit output format, and constraints, then validate what comes back. This exercise builds that pattern from scratch.

By the end, you will be able to:
- Design a structured prompt with role, instruction, and explicit output format
- Get an LLM to reliably return valid JSON matching a schema you define
- Test a prompt against multiple inputs and measure its actual accuracy, not just eyeball it

**Real-world context:** A bank's support inbox receives thousands of emails a day. Before any of them can be routed to the right team, someone (or something) has to read and categorise each one — this is exactly the kind of high-volume, well-defined classification task structured prompting is built for.

---

## 🧠 Concept Primer

### Why not just ask "what category is this email?"

You could — but you'd get inconsistent formatting back every time: sometimes a sentence, sometimes a label, sometimes an explanation you didn't ask for. A **structured prompt** fixes this by being explicit about three things:
1. **Role** — who the model should act as ("an email classification assistant for a bank")
2. **Task** — the exact decision to make, with the exact allowed answers listed out
3. **Output format** — the exact shape of the response, so your code can reliably parse it every time

### Why JSON specifically?

JSON is the universal "structured data" format your Python code can parse with one line (`json.loads`). Asking for JSON — and being explicit about the exact keys you want — turns "the model's opinion in prose" into "a value your program can use directly," which is the difference between a demo and a real pipeline.

### What is "confidence" doing in the output?

Asking the model to self-report a confidence score gives you a cheap signal for **triage**: low-confidence classifications can be routed to a human for review instead of being auto-processed, while high-confidence ones flow straight through. It's not a perfectly calibrated probability — but it's a useful, nearly-free heuristic.

---

## Step 1 — The dataset

`data/customer_support_emails.csv` — 8 realistic banking support emails, each with an `expected_category` column so this exercise can self-check accuracy, not just print results.

---

## Step 2 — Azure OpenAI setup

Needs one deployment (`AZURE_OPENAI_DEPLOYMENT`). See the package `README.md`.

---

## Step 3 — Code walkthrough

### Stage 1 — Define the categories and the structured system prompt

```python
CATEGORIES = ["fraud query", "account closure request", "loan inquiry", "fee dispute", "general complaint"]

SYSTEM_PROMPT = f"""You are an email classification assistant for a retail bank's customer support team.

Task: classify the customer email into EXACTLY ONE of these five categories:
{", ".join(CATEGORIES)}

Output format: return ONLY a JSON object with these exact keys, no explanation, no markdown formatting:
{{
  "category": "<one of the five categories above, exactly as written>",
  "confidence": <a number between 0.0 and 1.0>
}}
"""
```

**What this does:** the `CATEGORIES` list is defined once and interpolated into the prompt — if you need a 6th category later, you add it here and the prompt updates automatically. Notice the prompt explicitly says **"EXACTLY ONE"** and **"no explanation, no markdown formatting"** — both are there because, without them, models will often add a friendly sentence before the JSON, or wrap it in a ```` ```json ```` code fence, either of which would break a naive `json.loads()` call downstream.

### Stage 2 — Classify one email and parse the result

```python
def classify_email(client, deployment, email_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email_text},
    ]
    response = call_llm(client, deployment, messages, temperature=0)
    raw_output = response.choices[0].message.content

    try:
        parsed = json.loads(raw_output)
        return parsed.get("category"), parsed.get("confidence"), raw_output
    except json.JSONDecodeError:
        return None, None, raw_output
```

**What this does:** the system/user message split matters — the classification RULES live in the system message (stable across every email), and the actual email text is the user message (the only thing that changes per call). `temperature=0` keeps classification deterministic — the same email should always get the same category. The `try/except` guards against the model occasionally not returning perfectly valid JSON despite instructions — never assume LLM output is automatically well-formed.

### Stage 3 — Run against every email and self-check accuracy

```python
for _, row in df.iterrows():
    category, confidence, raw_output = classify_email(client, deployment, row["email_text"])
    is_correct = category == row["expected_category"]
    ...
```

**What this does:** because the dataset includes a human-labelled `expected_category` for every email, we can compute real accuracy instead of just reading outputs and guessing whether they look right — this is a small-scale preview of the systematic evaluation approach formalised in Module 6.

---

## Expected Output (illustrative)

```
EM001: Hi, I noticed a transaction on my account for $340 that I don't reco...
Expected:  fraud query
Predicted: fraud query (confidence: 0.95) ✅

EM005: Your mobile app has been crashing every time I try to log in for th...
Expected:  general complaint
Predicted: general complaint (confidence: 0.88) ✅

ACCURACY: 8/8 correct (100%)
```

Most emails in this dataset were written to be reasonably unambiguous, so high accuracy is expected — the homework asks you to write a genuinely ambiguous one.

---

## 🛠 Common Pitfalls

- **Forgetting `temperature=0`:** classification tasks want consistency; without it, the same email could get different categories on different runs.
- **Not handling JSON parse failures:** if you skip the `try/except`, one malformed response crashes your whole batch — always assume LLM output needs validation.
- **Categories with overlapping meaning:** if two categories could both reasonably apply to one email, expect lower confidence and possible misclassification — that's a prompt design problem, not a bug.

---

## 🏠 Homework Exercise

1. Add 2 new emails to `data/customer_support_emails.csv` — write at least one **deliberately ambiguous** email that could reasonably fit two categories (e.g., a fee dispute that's also somewhat a fraud concern).
2. Run the script and see what the model picks, and what confidence it reports.
3. Note whether the ambiguous email got a *lower* confidence score than the clear-cut ones — if it didn't, that's worth flagging as a prompt-design gap: should the prompt instruct the model to lower its confidence when a case is genuinely ambiguous?

**Hints:**
- Confidence scores are self-reported by the model, not mathematically derived — they can be miscalibrated. This homework is designed to help you feel that limitation directly.
- If you want to push further: add a 6th category to `CATEGORIES` and see if accuracy on the *existing* emails changes at all — a good category boundary shouldn't disturb classifications that were already unambiguous.

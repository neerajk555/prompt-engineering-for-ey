# Exercise 2 — Chain of Thought Prompting Comparison

**Difficulty:** Intermediate | **Time:** 15–20 min | **Theory link:** 3.2 Zero/One/Few-Shot Prompting / 3.3 Task Decomposition and Step-by-Step Prompting

---

## 🎯 Goal

Complex reasoning tasks — like a multi-step calculation — fail more often when a model is asked to jump straight to the answer than when it's asked to show its work first. This exercise proves that with a task that has a mathematically verifiable correct answer, so "did it work" isn't a matter of opinion.

By the end, you will be able to:
- Explain the difference between zero-shot, zero-shot Chain-of-Thought, and few-shot CoT prompting
- Predict which approach is more likely to get a multi-step calculation right, and why
- Measure prompting-technique accuracy objectively, against a known-correct answer

**Real-world context:** A bank needs its support chatbot to correctly calculate loan interest when asked — get this wrong even occasionally, and customers get incorrect financial information. This is exactly the kind of task where "sounds plausible" isn't good enough; it needs to be *right*.

---

## 🧠 Concept Primer

### The three approaches, in one sentence each

- **Zero-shot:** ask the question directly, no examples, no instruction to reason step by step — the model has to jump straight to an answer.
- **Zero-shot CoT:** same question, but add the magic phrase "think step by step" — this alone measurably improves accuracy on multi-step problems in many models, without giving a single example.
- **Few-shot CoT:** same "think step by step" instruction, PLUS one fully worked example showing exactly how to solve a similar problem — giving the model a template to follow, not just an instruction to follow.

### Why does "think step by step" actually help?

LLMs generate text one token at a time, and each new token is conditioned on everything generated before it — including the model's OWN prior output. If the model writes out intermediate steps, those steps become part of the context it uses to generate the next step, effectively giving it "working memory" on the page. Jumping straight to a final answer skips that scaffolding entirely, which is where arithmetic and multi-step logic errors creep in.

### Why does a worked example (few-shot) help beyond just "think step by step"?

An instruction like "think step by step" tells the model *to* reason, but not exactly *how* to structure that reasoning for this specific type of problem. A worked example shows the exact sequence of steps (convert units → apply formula → compute final answer) — reducing ambiguity about what "showing your work" should look like for this task specifically.

---

## Step 1 — The dataset

`data/loan_calculations.csv` — 5 loan scenarios with pre-computed, verified correct answers (simple interest calculations), so this exercise can grade itself.

---

## Step 2 — Azure OpenAI setup

Needs one deployment (`AZURE_OPENAI_DEPLOYMENT`). See the package `README.md`.

---

## Step 3 — Code walkthrough

### Stage 1 — Define the three prompt variants

```python
TASK_TEMPLATE = "A customer takes a loan of ${principal} at an annual interest rate of {rate}% for a tenure of {tenure} months, using simple interest. What is the total interest payable, and what is the total amount repayable?"

ZERO_SHOT_SUFFIX = "\n\nGive your final answer in exactly this format: 'Total interest: $X. Total repayable: $Y.'"
ZERO_SHOT_COT_SUFFIX = "\n\nThink step by step, showing your calculation, then give your final answer in exactly this format: 'Total interest: $X. Total repayable: $Y.'"
FEW_SHOT_COT_PREFIX = """Here is a worked example... [full worked example] ..."""
```

**What this does:** the underlying TASK is identical across all three approaches — same numbers, same question — only the *instruction wrapped around it* changes. This is the controlled-experiment discipline you've now seen in every module: change exactly one variable, hold everything else constant, so any difference in outcome is attributable to that one change.

### Stage 2 — Extract the numeric answer from free-form text

```python
def extract_numbers_from_answer(text):
    numbers = re.findall(r"\$?([\d,]+(?:\.\d+)?)", text)
    numbers = [float(n.replace(",", "")) for n in numbers]
    return numbers[-2:] if len(numbers) >= 2 else [None, None]
```

**What this does:** regardless of how much reasoning text precedes it, we've asked every approach to end with the same fixed format (`'Total interest: $X. Total repayable: $Y.'`), so a regex can reliably pull the LAST two dollar figures out of the response — those should be the final interest and repayable amounts, even if earlier numbers appeared during step-by-step reasoning.

### Stage 3 — Run all three approaches per scenario, grade against the known answer

```python
correct = (
    pred_interest is not None
    and abs(pred_interest - row["correct_total_interest"]) < 1.0
    and abs(pred_repayable - row["correct_total_repayable"]) < 1.0
)
```

**What this does:** grades each approach objectively against the pre-verified correct answer (with a small tolerance for rounding), rather than relying on a human reading the output and guessing whether it "looks right." This is what makes the comparison at the end meaningful instead of anecdotal.

---

## Expected Output (illustrative)

```
LC003: A customer takes a loan of $200000 at an annual interest rate of 7.5% for a tenure of 36 months...
Correct answer: interest=$45000.0, repayable=$245000.0

-- zero_shot -- ❌
Total interest: $45,000. Total repayable: $255000.

-- zero_shot_cot -- ✅
Step 1: Convert 36 months to 3 years.
Step 2: Interest = 200000 x 7.5 x 3 / 100 = 45000.
Step 3: Total repayable = 200000 + 45000 = 245000.
Total interest: $45000. Total repayable: $245000.

-- few_shot_cot -- ✅
...

ACCURACY BY APPROACH
zero_shot        60%
zero_shot_cot     100%
few_shot_cot      100%
```
*(Illustrative — the exact failure pattern will vary by model and run, but zero-shot is expected to be the least reliable of the three on multi-step arithmetic.)*

---

## 🛠 Common Pitfalls

- **Judging by "sounds confident" instead of the actual number:** a fluent wrong answer is still wrong — always check the extracted number against `correct_total_interest`/`correct_total_repayable`, don't eyeball it.
- **Forgetting the output-format instruction:** if a variant doesn't end with the fixed `'Total interest: $X...'` format, `extract_numbers_from_answer` may grab the wrong numbers — worth printing the raw answer alongside the extracted numbers when debugging.
- **Assuming CoT always wins:** on very simple tasks, zero-shot can match CoT — the accuracy gap widens specifically as task complexity increases, which is why this exercise uses a genuine multi-step calculation rather than a trivial one.

---

## 🏠 Homework Exercise

1. Add a 6th scenario to `data/loan_calculations.csv` with a **larger, more complex** tenure (e.g., 84 months) and rate with more decimal places (e.g., 11.35%) — compute the correct answer yourself first (simple interest = Principal × Rate × Years / 100) and verify it before adding it to the CSV.
2. Run all three approaches against it — does the accuracy gap between zero-shot and CoT widen or stay the same compared to the simpler scenarios?
3. Try changing `FEW_SHOT_COT_PREFIX`'s worked example to use a DIFFERENT calculation type (e.g., compound interest instead of simple interest) while keeping the task itself asking for simple interest — does the mismatched example confuse the model, or does it correctly generalise just the *reasoning structure* rather than blindly copying the specific numbers?

**Hints:**
- For question 3, this is testing whether few-shot examples teach *structure* or get *memorized* — a genuinely useful worked example should transfer the pattern (steps: convert units → apply formula → sum) even when the specific formula differs slightly.

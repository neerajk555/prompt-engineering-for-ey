"""
Exercise 4 — Prompt Versioning and A/B Testing
Module 3: Prompt Engineering

Goal: Write two distinct prompt variants for the same summarisation task,
run both against a test set of banking policy documents, score each output
against a simple rubric, and pick a winner with evidence.

Note: production teams typically use a dedicated tool like Promptfoo for this
(see README.md) — this exercise builds the same evaluation logic yourself in
plain Python first, so the underlying idea isn't a black box before you reach
for the tooling that automates it in Module 6.

Run from the project root: python exercises/04_prompt_ab_testing.py

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 04_Prompt_AB_Testing.md
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Variant A: minimal instruction, no format guidance
PROMPT_VARIANT_A = """Summarise this bank policy document for a customer-facing FAQ page:

{document}
"""

# Variant B: role, explicit constraints, explicit output format
PROMPT_VARIANT_B = """You are a customer communications specialist at a bank, rewriting internal policy
language into a clear, friendly FAQ answer for customers.

Rules:
- Maximum 2 sentences.
- No internal jargon (e.g. do not say "discretion of the fraud investigation team", say it plainly).
- End with a concrete, actionable takeaway for the customer if one exists in the source text.

Policy document:
{document}

Write the FAQ answer now.
"""

# A lightweight LLM-as-judge rubric, scored 1-5 on each dimension.
JUDGE_PROMPT_TEMPLATE = """Score the following FAQ answer against its source policy document on three dimensions,
each from 1 (poor) to 5 (excellent). Return ONLY a JSON object with these exact keys:
{{"clarity": <1-5>, "completeness": <1-5>, "conciseness": <1-5>}}

Source policy document:
{document}

FAQ answer to score:
{answer}
"""


def load_data(path="data/policy_documents.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} policy documents")
    return df


def get_client_and_deployment():
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(
            f"\n[WARNING] Azure OpenAI not configured — missing: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill it in (see README.md). Exiting.\n"
        )
        return None, None

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    return client, deployment


def call_llm(client, deployment, messages, temperature=None):
    try:
        if temperature is not None:
            return client.chat.completions.create(model=deployment, messages=messages, temperature=temperature)
        return client.chat.completions.create(model=deployment, messages=messages)
    except Exception as e:
        if temperature is not None and "temperature" in str(e).lower():
            return client.chat.completions.create(model=deployment, messages=messages)
        raise


def generate_answer(client, deployment, prompt_template, document_text):
    prompt = prompt_template.format(document=document_text)
    response = call_llm(client, deployment, [{"role": "user", "content": prompt}], temperature=0.3)
    return response.choices[0].message.content


def judge_answer(client, deployment, document_text, answer_text):
    import json
    prompt = JUDGE_PROMPT_TEMPLATE.format(document=document_text, answer=answer_text)
    response = call_llm(client, deployment, [{"role": "user", "content": prompt}], temperature=0)
    raw = response.choices[0].message.content
    try:
        scores = json.loads(raw)
        return scores.get("clarity"), scores.get("completeness"), scores.get("conciseness")
    except json.JSONDecodeError:
        return None, None, None


def main():
    df = load_data()
    client, deployment = get_client_and_deployment()
    if client is None:
        return

    results = []

    for _, row in df.iterrows():
        print(f"\n{'=' * 80}\n{row['doc_id']}: {row['document_text'][:80]}...")

        answer_a = generate_answer(client, deployment, PROMPT_VARIANT_A, row["document_text"])
        answer_b = generate_answer(client, deployment, PROMPT_VARIANT_B, row["document_text"])

        clarity_a, complete_a, concise_a = judge_answer(client, deployment, row["document_text"], answer_a)
        clarity_b, complete_b, concise_b = judge_answer(client, deployment, row["document_text"], answer_b)

        print(f"\n-- Variant A --\n{answer_a}\nScores: clarity={clarity_a} completeness={complete_a} conciseness={concise_a}")
        print(f"\n-- Variant B --\n{answer_b}\nScores: clarity={clarity_b} completeness={complete_b} conciseness={concise_b}")

        results.append({"doc_id": row["doc_id"], "variant": "A", "clarity": clarity_a, "completeness": complete_a, "conciseness": concise_a})
        results.append({"doc_id": row["doc_id"], "variant": "B", "clarity": clarity_b, "completeness": complete_b, "conciseness": concise_b})

    results_df = pd.DataFrame(results)
    results_df["total_score"] = results_df[["clarity", "completeness", "conciseness"]].sum(axis=1)
    results_df.to_csv("prompt_ab_test_output.csv", index=False)

    print(f"\n{'=' * 80}\nAVERAGE SCORES BY VARIANT")
    print(results_df.groupby("variant")[["clarity", "completeness", "conciseness", "total_score"]].mean().round(2))
    print("\nSaved full results to prompt_ab_test_output.csv")


if __name__ == "__main__":
    main()

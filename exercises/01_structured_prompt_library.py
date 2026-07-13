"""
Exercise 1 — Build a Structured Prompt Library for a Business Use Case
Module 3: Prompt Engineering

Goal: Design a reusable, structured prompt that classifies incoming banking
support emails into one of five categories, returning a category label and
confidence score as validated JSON. Test against 8 real-style emails and
check accuracy against expected labels.

Run from the project root: python exercises/01_structured_prompt_library.py

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 01_Structured_Prompt_Library.md
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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


def load_data(path="data/customer_support_emails.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} customer support emails")
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
    """Tries the call with temperature if given; automatically retries without
    it if the deployment is a reasoning model that rejects the parameter."""
    try:
        if temperature is not None:
            return client.chat.completions.create(model=deployment, messages=messages, temperature=temperature)
        return client.chat.completions.create(model=deployment, messages=messages)
    except Exception as e:
        if temperature is not None and "temperature" in str(e).lower():
            return client.chat.completions.create(model=deployment, messages=messages)
        raise


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


def main():
    df = load_data()
    client, deployment = get_client_and_deployment()
    if client is None:
        return

    results = []
    correct_count = 0

    for _, row in df.iterrows():
        category, confidence, raw_output = classify_email(client, deployment, row["email_text"])
        is_correct = category == row["expected_category"]
        correct_count += int(is_correct)

        results.append({
            "email_id": row["email_id"],
            "expected_category": row["expected_category"],
            "predicted_category": category,
            "confidence": confidence,
            "correct": is_correct,
        })

        match_symbol = "✅" if is_correct else "❌"
        print(f"\n{'=' * 80}\n{row['email_id']}: {row['email_text'][:70]}...")
        print(f"Expected:  {row['expected_category']}")
        print(f"Predicted: {category} (confidence: {confidence}) {match_symbol}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("classification_results_output.csv", index=False)

    accuracy = correct_count / len(df) * 100
    print(f"\n{'=' * 80}")
    print(f"ACCURACY: {correct_count}/{len(df)} correct ({accuracy:.0f}%)")
    print(f"Saved full results to classification_results_output.csv")


if __name__ == "__main__":
    main()

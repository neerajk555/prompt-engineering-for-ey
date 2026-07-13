"""
Exercise 2 — Chain of Thought Prompting Comparison
Module 3: Prompt Engineering

Goal: Run the same loan-interest-calculation task through three prompting
approaches — zero-shot, zero-shot CoT ("think step by step"), and few-shot
CoT — and compare accuracy against a known-correct answer for each.

Run from the project root: python exercises/02_chain_of_thought.py

Full concept explanation, line-by-line walkthrough, expected output, and homework
are in 02_Chain_of_Thought.md
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TASK_TEMPLATE = "A customer takes a loan of ${principal} at an annual interest rate of {rate}% for a tenure of {tenure} months, using simple interest. What is the total interest payable, and what is the total amount repayable?"

ZERO_SHOT_SUFFIX = "\n\nGive your final answer in exactly this format: 'Total interest: $X. Total repayable: $Y.'"

ZERO_SHOT_COT_SUFFIX = "\n\nThink step by step, showing your calculation, then give your final answer in exactly this format: 'Total interest: $X. Total repayable: $Y.'"

FEW_SHOT_COT_PREFIX = """Here is a worked example of how to solve this kind of problem:

Example:
A customer takes a loan of $10,000 at an annual interest rate of 6% for a tenure of 12 months, using simple interest. What is the total interest payable, and what is the total amount repayable?

Step 1: Convert tenure to years: 12 months = 1 year.
Step 2: Apply the simple interest formula: Interest = Principal x Rate x Time / 100 = 10000 x 6 x 1 / 100 = 600.
Step 3: Total repayable = Principal + Interest = 10000 + 600 = 10600.
Total interest: $600. Total repayable: $10600.

Now solve this new problem the same way, showing your steps, then give your final answer in exactly this format: 'Total interest: $X. Total repayable: $Y.'

"""


def load_data(path="data/loan_calculations.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} loan calculation scenarios")
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


def extract_numbers_from_answer(text):
    """Pulls the two dollar figures out of a 'Total interest: $X. Total repayable: $Y.' style answer."""
    # NOTE: the pattern requires a leading digit (\d) before allowing further
    # digits/commas. An earlier version used [\d,]+ alone, which could match
    # a lone stray comma in ordinary prose (e.g. "for example, ...") with no
    # digit at all, producing an empty string after removing commas and
    # crashing on float(''). Requiring a leading digit closes that gap.
    numbers = re.findall(r"\$?(\d[\d,]*(?:\.\d+)?)", text)
    numbers = [float(n.replace(",", "")) for n in numbers if n.replace(",", "") != ""]
    return numbers[-2:] if len(numbers) >= 2 else [None, None]


def run_approach(client, deployment, task_text, approach):
    if approach == "zero_shot":
        prompt = task_text + ZERO_SHOT_SUFFIX
    elif approach == "zero_shot_cot":
        prompt = task_text + ZERO_SHOT_COT_SUFFIX
    elif approach == "few_shot_cot":
        prompt = FEW_SHOT_COT_PREFIX + task_text
    else:
        raise ValueError(approach)

    response = call_llm(client, deployment, [{"role": "user", "content": prompt}], temperature=0)
    answer_text = response.choices[0].message.content
    predicted_interest, predicted_repayable = extract_numbers_from_answer(answer_text)
    return answer_text, predicted_interest, predicted_repayable


def main():
    df = load_data()
    client, deployment = get_client_and_deployment()
    if client is None:
        return

    approaches = ["zero_shot", "zero_shot_cot", "few_shot_cot"]
    results = []

    for _, row in df.iterrows():
        task_text = TASK_TEMPLATE.format(
            principal=int(row["principal"]), rate=row["annual_rate_percent"], tenure=int(row["tenure_months"])
        )
        print(f"\n{'=' * 80}\n{row['scenario_id']}: {task_text}")
        print(f"Correct answer: interest=${row['correct_total_interest']}, repayable=${row['correct_total_repayable']}")

        for approach in approaches:
            answer_text, pred_interest, pred_repayable = run_approach(client, deployment, task_text, approach)
            correct = (
                pred_interest is not None
                and abs(pred_interest - row["correct_total_interest"]) < 1.0
                and abs(pred_repayable - row["correct_total_repayable"]) < 1.0
            )
            results.append({
                "scenario_id": row["scenario_id"],
                "approach": approach,
                "predicted_interest": pred_interest,
                "predicted_repayable": pred_repayable,
                "correct": correct,
            })
            symbol = "✅" if correct else "❌"
            print(f"\n-- {approach} -- {symbol}")
            print(answer_text[:300])

    results_df = pd.DataFrame(results)
    results_df.to_csv("cot_comparison_output.csv", index=False)

    print(f"\n{'=' * 80}\nACCURACY BY APPROACH")
    print(results_df.groupby("approach")["correct"].mean().mul(100).round(0).astype(int).astype(str) + "%")
    print("\nSaved full results to cot_comparison_output.csv")


if __name__ == "__main__":
    main()
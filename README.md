# Module 3 — Prompt Engineering
## Hands-On Lab Package

**Days 4–6 | Intermediate | 4 exercises, ~15–20 min each**

Every exercise in this module calls Azure OpenAI — Module 3 is about the craft of prompt design itself, so every lesson needs a real model to observe. **Set up Azure OpenAI before the session.**

```
module3_labs/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── customer_support_emails.csv    (Exercise 1)
│   ├── loan_calculations.csv          (Exercise 2)
│   ├── chatbot_system_prompt.csv      (Exercise 3)
│   ├── injection_attempts.csv         (Exercise 3)
│   └── policy_documents.csv           (Exercise 4)
└── exercises/
    ├── 01_Structured_Prompt_Library.md      + 01_structured_prompt_library.py
    ├── 02_Chain_of_Thought.md               + 02_chain_of_thought.py
    ├── 03_Prompt_Injection_Defence.md       + 03_prompt_injection_defence.py
    └── 04_Prompt_AB_Testing.md              + 04_prompt_ab_testing.py
```

| # | Exercise | Theory Topic | Needs |
|---|----------|--------------|-------|
| 1 | Structured Prompt Library | 3.1 Prompt Structure / 3.4 Structured Output Prompting | 1 deployment |
| 2 | Chain-of-Thought Prompting Comparison | 3.2 Zero/Few-Shot / 3.3 Task Decomposition & CoT | 1 deployment |
| 3 | Prompt Injection Attack & Defence | 3.9 Prompt Injection Risks and Defensive Prompting | 1 deployment |
| 4 | Prompt Versioning & A/B Testing | 3.10 Prompt Evaluation, Versioning, Reusable Libraries | 1 deployment |

All four exercises share **one** deployment (`AZURE_OPENAI_DEPLOYMENT`) — same single-variable pattern as Module 1's Exercise 4, simpler than Module 2's two-deployment setup.

---

## Quick Start (TL;DR)

```bash
python -m venv module3-env && source module3-env/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your Azure OpenAI credentials — see below
python exercises/01_structured_prompt_library.py
```

---

## One-Time Environment Setup

### 1. Create and activate a virtual environment
```bash
python -m venv module3-env
source module3-env/bin/activate        # Windows: module3-env\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Azure OpenAI

If you already have a resource from Module 1 or 2, you can reuse it — you only need **one** deployment for this module.

**3.1 — Create the resource** (skip if you already have one):
- [Azure Portal](https://portal.azure.com) → search **"Azure OpenAI"** → **Create**
- Fill in Subscription / Resource group / Region / Name / Pricing tier (Standard S0)
- **Review + Create** → **Create** → **Go to resource**

**3.2 — Check what's actually deployable, then deploy it:**

> ⚠️ **Azure's available models have been changing rapidly through 2026.** Don't assume any specific model name from this or any guide is currently deployable — check live.
1. Resource → **Go to Azure AI Foundry portal** → **Deployments** → **+ Create new deployment**
2. Filter **Deployment type** to **Standard**, see what's actually selectable
3. Pick any available chat model — **Deployment name:** e.g. `prompt-eng-deploy`
4. Wait for **"Succeeded"**

**A note on reasoning vs non-reasoning models:** every script in this module that sets `temperature` does so with an automatic fallback — if your deployment is a reasoning model that rejects `temperature`, the script detects the specific error and silently retries without it. You don't need to do anything differently either way.

**3.3 — Get credentials:** resource → **Keys and Endpoint** → copy **KEY 1** and the **Endpoint**.

**3.4 — Configure `.env`:**
```bash
cp .env.example .env
```
```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=prompt-eng-deploy
AZURE_OPENAI_API_VERSION=2024-10-21
```

---

## Running the Exercises

Run from the **project root** (`module3_labs/`), not from inside `exercises/`:

```bash
python exercises/01_structured_prompt_library.py
python exercises/02_chain_of_thought.py
python exercises/03_prompt_injection_defence.py
python exercises/04_prompt_ab_testing.py
```

If Azure OpenAI isn't configured yet, each script prints a clear warning and exits cleanly rather than crashing.

---

## Editing the Datasets

Every script reads its data from `data/` — nothing is hardcoded in the Python files. Add a new email, a new loan scenario, a new injection attempt, or a new policy document by editing the relevant CSV, then re-run. No code changes needed.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `[WARNING] Azure OpenAI not configured — missing: ...` | Complete Step 3 above — `.env` isn't filled in yet. |
| `AuthenticationError` / `401` | Check `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` are copied exactly from **Keys and Endpoint**, no extra spaces or quotes. |
| `ServiceModelDeprecating` when creating a deployment | The model you picked can't be newly deployed anymore. Go back to Azure AI Foundry, filter to **Standard**, and pick from whatever's actually selectable right now. |
| `BadRequestError` mentioning `temperature` | Should be handled automatically by every script in this module — if you still see this, you may be running an older copy; re-download the latest version. |
| `NotFoundError` / `404` referencing the model name | You're passing the **model name** instead of the **deployment name**. `AZURE_OPENAI_DEPLOYMENT` must match the deployment name *you chose*, not the underlying model. |
| `RateLimitError` / `429` | Hit your deployment's quota — common with a full class hitting a shared resource. Wait a minute and retry, or check quota under **Quotas** in Azure AI Foundry. |
| `ModuleNotFoundError` for any package | Confirm your virtual environment is activated, then re-run `pip install -r requirements.txt`. |

---

## Suggested Pacing (fits inside the Days 4–6 sessions)

| Time | Activity |
|------|----------|
| 0–5 min | Environment + `.env` check |
| 5–25 min | Exercise 1 — Structured Prompt Library |
| 25–45 min | Exercise 2 — Chain-of-Thought Comparison |
| 45–65 min | Exercise 3 — Prompt Injection & Defence |
| 65–85 min | Exercise 4 — Prompt A/B Testing |

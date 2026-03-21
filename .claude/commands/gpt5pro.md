# GPT-5-Pro - Azure OpenAI Reasoning Model Review

Call GPT-5-Pro (reasoning model) via Azure OpenAI Responses API to review, analyze, or generate content.

## Arguments

`$ARGUMENTS` — Task and target. Examples: `review §1-§4`, `review <file>`, `ask "<question>" --files §1 §3`

## Protocol

### Step 1: Determine Task

Parse `$ARGUMENTS` → task type (`review` | `ask`), target (sections/files/question), flags (`--files`, `--max-tokens`).

### Step 2: Gather Context

- `review`: Read target .tex files + `docs/paper_state/*/framing.md` for locked terminology
- `ask`: Read any `--files`, pass question directly

### Step 3: Build and Execute API Call

Read `.claude/resources/azure_models.md` for API configuration and Python template. Key points:
- **URL**: `/openai/responses?api-version=2025-03-01-preview` (NOT `/deployments/...`)
- **Model in body**: `"model": "gpt-5-pro"`
- **Token budget**: `max_output_tokens >= 16000` (reasoning uses ~8-10k)

For review tasks, read `.claude/resources/review_prompt_template.md` for the prompt structure.

Write script to `/tmp/gpt5pro_call.py` and execute with `timeout=300`.

### Step 4: Process Results

1. Display output to user
2. For `review`: evaluate each suggestion against framing.md constraints
3. Summarize: N total, M recommended

## Begin

Parse `$ARGUMENTS` and execute.

# Azure OpenAI Model Reference

## Endpoint Priority (3-Tier Fallback)

All API calls (chat completions AND responses) try endpoints in priority order. If one fails, the next is attempted automatically.

| Priority | Name | Endpoint | Key Env Var | Notes |
|----------|------|----------|-------------|-------|
| 1 | AI Foundry | `aorc-8683-resource.services.ai.azure.com` | `AZURE_AI_FOUNDRY_KEY` | Azure AI Services, model in body |
| 2 | Sweden Central | `aorc-mlyilk23-swedencentral.cognitiveservices.azure.com` | `AZURE_OPENAI_SWEDEN_KEY` | gpt-5-nano deployment |
| 3 | East US 2 (fallback) | `85409-mk9jw7uo-eastus2.openai.azure.com` | `AZURE_OPENAI_API_KEY` | Original, all models |

---

## GPT-5-Pro (Reasoning Model)

```yaml
API:   Responses API (/openai/responses), NOT /chat/completions
Model: gpt-5-pro (in request body, NOT in URL path)
Fallback: AI Foundry → Sweden Central → East US 2 (same 3-tier priority)
```

### Implementation Notes

1. **Responses API only** — Does not support `/chat/completions`. Use `/openai/responses`.
2. **Model in body** — `"model": "gpt-5-pro"` in JSON body.
3. **Token budget** — Reasoning uses ~8-10k tokens. Set `max_output_tokens >= 16000`.
4. **No effort control** — Only `reasoning.effort: "high"` supported.
5. **No temperature** — Do not pass `temperature`.
6. **Timeout** — 30-120s typical. Set 300s.
7. **3-tier fallback** — Same priority as chat completions: AI Foundry → Sweden Central → East US 2.

### Error Reference

| Error | Cause | Fix |
|-------|-------|-----|
| 404 on `/deployments/gpt-5-pro/responses` | Wrong URL | Use `/openai/responses` without deployment path |
| 400 "chatCompletion not supported" | Wrong API | Use Responses API |
| 400 "medium not supported" | Effort control | Remove or set to `"high"` |
| 200 but empty output | Budget too low | `max_output_tokens >= 16000` |
| 200 status "incomplete" | Truncated | Increase `max_output_tokens` |

### Python Template (GPT-5-Pro with 3-Tier Fallback)

```python
import os, json, requests

def gpt5pro_responses(prompt, model="gpt-5-pro", max_output_tokens=16000, timeout=300):
    """Call Azure Responses API with 3-tier fallback.

    Priority: AI Foundry → Sweden Central → East US 2.
    """
    endpoints = [
        # Priority 1: AI Foundry
        {
            "url": "https://aorc-8683-resource.services.ai.azure.com/openai/responses?api-version=2025-03-01-preview",
            "key_env": "AZURE_AI_FOUNDRY_KEY",
            "name": "AI Foundry",
        },
        # Priority 2: Sweden Central
        {
            "url": "https://aorc-mlyilk23-swedencentral.cognitiveservices.azure.com/openai/responses?api-version=2025-03-01-preview",
            "key_env": "AZURE_OPENAI_SWEDEN_KEY",
            "name": "Sweden Central",
        },
        # Priority 3: East US 2 (fallback)
        {
            "url": "https://85409-mk9jw7uo-eastus2.openai.azure.com/openai/responses?api-version=2025-03-01-preview",
            "key_env": "AZURE_OPENAI_API_KEY",
            "name": "East US 2",
        },
    ]

    body = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}

    errors = []
    for ep in endpoints:
        key = os.environ.get(ep["key_env"])
        if not key:
            errors.append(f"{ep['name']}: {ep['key_env']} not set")
            continue
        try:
            resp = requests.post(ep["url"],
                headers={"api-key": key, "Content-Type": "application/json"},
                json=body, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                text += c["text"]
                usage = data.get("usage", {})
                reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
                output_tokens = usage.get("output_tokens", 0) - reasoning
                print(f"[via {ep['name']}]")
                print(f"[Usage: {usage.get('input_tokens',0)} in / {reasoning} reasoning / {output_tokens} output]")
                return text
            errors.append(f"{ep['name']}: HTTP {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            errors.append(f"{ep['name']}: {e}")

    print("All endpoints failed:")
    for e in errors:
        print(f"  - {e}")
    return None


# Usage:
result = gpt5pro_responses(prompt)
print(result)
```

---

## Chat Completions (with 3-Tier Fallback)

### Endpoint Details

**Priority 1 — AI Foundry (`aorc-8683`)**
```
URL: https://aorc-8683-resource.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview
Auth: api-key header with AZURE_AI_FOUNDRY_KEY
Model: specified in request body
```

**Priority 2 — Sweden Central (`gpt-5-nano`)**
```
URL: https://aorc-mlyilk23-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-01-01-preview
Auth: api-key header with AZURE_OPENAI_SWEDEN_KEY
Model: gpt-5-nano (in URL path)
```

**Priority 3 — East US 2 (fallback, all models)**
```
URL: https://85409-mk9jw7uo-eastus2.openai.azure.com/openai/deployments/{deploy}/chat/completions?api-version=2024-12-01-preview
Auth: api-key header with AZURE_OPENAI_API_KEY
```

| Model | Deployment | Endpoint |
|-------|-----------|----------|
| gpt-5.2-chat | `gpt-5.2-chat` | East US 2 only |
| gpt-5-nano | `gpt-5-nano` | Sweden Central |
| gpt-5-mini | `gpt-5-mini` | East US 2 only |
| gpt-4.1 | `gpt-4.1` | East US 2 only |
| gpt-4.1-mini | `gpt-4.1-mini` | East US 2 only |
| o4-mini | `o4-mini` | East US 2 only |
| o1 | `o1` | East US 2 only |

### Python Template (Chat Completions with Fallback)

```python
import os, requests

def chat_completions(prompt, model=None, max_tokens=4000, timeout=120):
    """Call Azure chat completions with 3-tier fallback.

    Priority: AI Foundry → Sweden Central (gpt-5-nano) → East US 2 (fallback).
    """
    messages = [{"role": "user", "content": prompt}]

    endpoints = [
        # Priority 1: AI Foundry
        {
            "url": "https://aorc-8683-resource.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
            "key_env": "AZURE_AI_FOUNDRY_KEY",
            "body": {"messages": messages, "max_tokens": max_tokens},
            "name": "AI Foundry",
        },
        # Priority 2: Sweden Central (gpt-5-nano)
        {
            "url": "https://aorc-mlyilk23-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-01-01-preview",
            "key_env": "AZURE_OPENAI_SWEDEN_KEY",
            "body": {"messages": messages, "max_tokens": max_tokens},
            "name": "Sweden Central",
        },
        # Priority 3: East US 2 (fallback, supports all models)
        {
            "url": f"https://85409-mk9jw7uo-eastus2.openai.azure.com/openai/deployments/{model or 'gpt-5.2-chat'}/chat/completions?api-version=2024-12-01-preview",
            "key_env": "AZURE_OPENAI_API_KEY",
            "body": {"messages": messages, "max_completion_tokens": max_tokens},
            "name": "East US 2",
        },
    ]

    # If model specified in body for Foundry endpoint
    if model:
        endpoints[0]["body"]["model"] = model

    errors = []
    for ep in endpoints:
        key = os.environ.get(ep["key_env"])
        if not key:
            errors.append(f"{ep['name']}: {ep['key_env']} not set")
            continue
        try:
            resp = requests.post(ep["url"],
                headers={"api-key": key, "Content-Type": "application/json"},
                json=ep["body"], timeout=timeout)
            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"]
                print(f"[via {ep['name']}]")
                return result
            errors.append(f"{ep['name']}: HTTP {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            errors.append(f"{ep['name']}: {e}")

    print("All endpoints failed:")
    for e in errors:
        print(f"  - {e}")
    return None


# Usage:
result = chat_completions(prompt)  # auto-fallback, default model
result = chat_completions(prompt, model="gpt-5.2-chat")  # specific model (East US 2 fallback)
```

# Custom Model Routing with `config.yaml`

g4f supports a `config.yaml` file that lets you define **custom model routes** – named
models that are transparently forwarded to one or more real providers based on
availability, quota balance, and recent error counts.

This is similar to the [LiteLLM](https://docs.litellm.ai/) routing configuration.

---

## Quick start

1. Place a `config.yaml` file in the same directory as your `.har` / `.json`
   cookie files (the "cookies dir").
   * Default location: `~/.config/g4f/cookies/config.yaml`
   * Alternative: `./har_and_cookies/config.yaml`

2. Define your routes (see format below).

3. g4f loads the file automatically when it reads the cookie directory
   (e.g. on API server start-up, or when `read_cookie_files()` is called).

4. Request the custom model name from any client:

```python
from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="my-gpt4",   # defined in config.yaml
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## File format

```yaml
models:
  - name: "<model-name>"          # the name clients use
    providers:
      - provider: "<ProviderName>"  # g4f provider class name
        model: "<provider-model>"   # model name passed to that provider
        condition: "<expression>"   # optional – see below
      - provider: "..."             # fallback provider (no condition = always eligible)
        model: "..."
```

### Keys

| Key | Required | Description |
|-----|----------|-------------|
| `name` | ✅ | The model name used by clients. |
| `providers` | ✅ | Ordered list of provider candidates. |
| `provider` | ✅ | Provider class name (e.g. `"OpenaiAccount"`, `"PollinationsAI"`). |
| `model` | | Model name forwarded to the provider. Defaults to the route `name`. |
| `condition` | | Boolean expression controlling when this provider is eligible. |

---

## Condition expressions

The `condition` field is a boolean expression evaluated before each request.
It can reference the following variables:

### `quota` – full provider quota dict

Each provider that implements `get_quota()` returns a **provider-specific** dict.
The result is cached in memory (5 min TTL) and invalidated on 429 responses.

Access any field with **dot-notation**:

| Provider | `get_quota()` format | Example condition |
|----------|---------------------|-------------------|
| `PollinationsAI` | `{"balance": float}` | `quota.balance > 0` |
| `Yupp` | `{"credits": {"remaining": int, "total": int}}` | `quota.credits.remaining > 100` |
| `PuterJS` | raw metering JSON from the API | `quota.total_requests < 1000` |
| `GeminiCLI` | `{"buckets": [...]}` | `error_count < 3` |
| `GithubCopilot` | usage details dict | `error_count < 5` |

Missing keys resolve to `0.0` (no error raised).

### `balance` – shorthand alias

`balance` is a convenience shorthand for `quota.balance`.  It is preserved for
backward compatibility and is most useful with **PollinationsAI** which returns
`{"balance": float}`.  For other providers, prefer the explicit `quota.*` form.

### `error_count`

Number of errors recorded for this provider in the last **1 hour**.  Errors
older than 1 hour are automatically pruned.

### Operators

| Operator | Meaning |
|----------|---------|
| `>` `<` `>=` `<=` | Numeric comparison |
| `==` `!=` | Equality / inequality |
| `and` `or` `not` | Logical connectives |
| `(` `)` | Grouping |

### Examples

```yaml
# PollinationsAI – uses quota.balance shorthand
condition: "balance > 0"
condition: "balance > 0 or error_count < 3"

# Yupp – provider-specific nested field
condition: "quota.credits.remaining > 0"
condition: "quota.credits.remaining > 0 or error_count < 3"

# Any provider – error-count-only conditions work universally
condition: "error_count < 3"
condition: "error_count == 0"
```

When the condition is **absent** or evaluates to `True`, the provider is
eligible.  When it evaluates to `False` the provider is skipped and g4f
tries the next one in the list.

---

## Quota caching

Quota values are fetched via the provider's `get_quota()` method and cached in
memory for **5 minutes** (configurable via `QuotaCache.ttl`).

When a provider returns an HTTP **429 (Too Many Requests)** error the cache
entry for that provider is **immediately invalidated**, so the next routing
decision fetches a fresh quota value before deciding.

---

## Error counting

Every time a provider raises an exception the error counter for that provider
is incremented.  Errors older than **1 hour** are automatically pruned.

Reference `error_count` in a condition to avoid retrying providers that have
been failing repeatedly.

---

## Full example

```yaml
# ~/.config/g4f/cookies/config.yaml

models:
  # PollinationsAI: use quota.balance shorthand
  - name: "my-gpt4"
    providers:
      - provider: "OpenaiAccount"
        model: "gpt-4o"
        condition: "balance > 0 or error_count < 3"
      - provider: "PollinationsAI"
        model: "openai-large"

  # Yupp: provider-specific nested quota field
  - name: "yupp-chat"
    providers:
      - provider: "Yupp"
        model: "gpt-4o"
        condition: "quota.credits.remaining > 0 or error_count < 3"
      - provider: "PollinationsAI"
        model: "openai-large"

  # Universal: error-count-only condition works for any provider
  - name: "llama-fast"
    providers:
      - provider: "Groq"
        model: "llama-3.3-70b"
        condition: "error_count < 3"
      - provider: "DeepInfra"
        model: "meta-llama/Llama-3.3-70B-Instruct"
```

---

## Python API

The routing machinery is exposed in `g4f.providers.config_provider`:

```python
from g4f.providers.config_provider import (
    RouterConfig,        # load / query routes
    QuotaCache,          # inspect / invalidate quota cache
    ErrorCounter,        # inspect / reset error counters
    evaluate_condition,  # evaluate a condition string directly
)

# Reload routes from a custom path
RouterConfig.load("/path/to/config.yaml")

# Check if a route exists
route = RouterConfig.get("my-gpt4")  # returns ModelRouteConfig or None

# Manually invalidate quota cache (e.g. after detecting 429)
QuotaCache.invalidate("OpenaiAccount")

# Check error count
count = ErrorCounter.get_count("OpenaiAccount")

# Evaluate a condition string with a full provider-specific quota dict
# (PollinationsAI)
ok = evaluate_condition("balance > 0 or error_count < 3", {"balance": 0.0}, 2)
# True

# Yupp-style nested quota
ok = evaluate_condition(
    "quota.credits.remaining > 0",
    {"credits": {"remaining": 500, "total": 5000}},
    0,
)
# True
```

---

## Requirements

PyYAML must be installed:

```bash
pip install pyyaml
```

It is included in the full `requirements.txt`.  If PyYAML is absent g4f logs
a warning and skips config.yaml loading.

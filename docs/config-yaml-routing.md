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
It can reference two variables:

| Variable | Type | Description |
|----------|------|-------------|
| `balance` | `float` | Provider quota balance, fetched via `get_quota()` and **cached** for 5 minutes. Returns `0.0` if the provider has no `get_quota` method or the call fails. |
| `error_count` | `int` | Number of errors recorded for this provider in the last **1 hour**. |
| `get_quota.balance` | `float` | Alias for `balance`. |

### Operators

| Operator | Meaning |
|----------|---------|
| `>` `<` `>=` `<=` | Numeric comparison |
| `==` `!=` | Equality / inequality |
| `and` `or` `not` | Logical connectives |
| `(` `)` | Grouping |

### Examples

```yaml
condition: "balance > 0"
condition: "error_count < 3"
condition: "balance > 0 or error_count < 3"
condition: "balance >= 10 and error_count == 0"
condition: "(balance > 0 or error_count < 5) and error_count < 10"
```

When the condition is **absent** or evaluates to `True`, the provider is
eligible.  When it evaluates to `False` the provider is skipped and g4f
tries the next one in the list.

---

## Quota caching

Quota values (`balance`) are fetched via the provider's `get_quota()` method
and cached in memory for **5 minutes** (configurable via
`QuotaCache.ttl`).

When a provider returns an HTTP **429 (Too Many Requests)** error the cache
entry for that provider is **immediately invalidated**, so the next routing
decision fetches a fresh balance before deciding.

---

## Error counting

Every time a provider raises an exception the error counter for that provider
is incremented.  Errors older than **1 hour** are automatically pruned.

You can reference `error_count` in a condition to avoid retrying providers
that have been failing repeatedly.

---

## Full example

```yaml
# ~/.config/g4f/cookies/config.yaml

models:
  # Prefer OpenaiAccount when it has quota; fall back to PollinationsAI.
  - name: "my-gpt4"
    providers:
      - provider: "OpenaiAccount"
        model: "gpt-4o"
        condition: "balance > 0 or error_count < 3"
      - provider: "PollinationsAI"
        model: "openai-large"

  # Simple two-provider fallback, no conditions.
  - name: "fast-chat"
    providers:
      - provider: "PollinationsAI"
        model: "openai"
      - provider: "Gemini"
        model: "gemini-2.0-flash"

  # Only use Groq when it has not exceeded 3 recent errors.
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
    RouterConfig,   # load / query routes
    QuotaCache,     # inspect / invalidate quota cache
    ErrorCounter,   # inspect / reset error counters
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

# Evaluate a condition string
ok = evaluate_condition("balance > 0 or error_count < 3", balance=0.0, error_count=2)
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

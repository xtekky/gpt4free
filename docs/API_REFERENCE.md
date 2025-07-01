# g4f API Reference

> This document gives a **human-curated** overview of all *public* classes, functions and constants in the `g4f` package.  Internal helpers (modules prefixed with an underscore or imported only for backward-compatibility) are intentionally omitted.

For detailed type information inspect the inline type hints or open the corresponding source files in your editor.

---

## Package-level exports (`import g4f`)

| Symbol | Type | Description |
| ------ | ---- | ----------- |
| `ChatCompletion` | `class` | High-level static interface for creating chat (and optionally image) completions. Mirrors the official OpenAI semantics. |
| `Model` | `dataclass` | Immutable description of a model + its preferred provider. Registered on import. |
| `ModelRegistry` | `class` | Global registry; look-up utility for `Model` instances and aliases. |
| `Client` | `class` | Synchronous convenience wrapper combining chat & image endpoints. |
| `AsyncClient` | `class` | Asynchronous variant of `Client`. |
| `get_cookies / set_cookies` | `function` | Persist and retrieve provider-specific cookies used during web-scraping. |

---

## 1. `ChatCompletion`

```python
class ChatCompletion:
    @staticmethod
    def create(
        model: Union[Model, str],
        messages: Messages,
        provider: Union[ProviderType, str, None] = None,
        stream: bool = False,
        image: ImageType | None = None,
        image_name: str | None = None,
        ignore_working: bool = False,
        ignore_stream: bool = False,
        **provider_kwargs,
    ) -> str | Iterator[ChatCompletionChunk] | ChatCompletionChunk:
        """Generate a completion. If *stream* is *True* an iterator of chunks is returned."""

    @staticmethod
    async def create_async(...):
        """Asynchronous mirror of `create`. Returns either a Coroutine (non-stream) or an async iterator (stream)."""
```

### Behaviour

1. **Automatic provider routing** – When *provider* is `None` the best provider for the chosen *model* is selected via `models.py`.
2. **Proxies** – honours `G4F_PROXY` env variable if `proxy` kwarg is omitted.
3. **Images** – pass a file-like object or bytes via the `image` parameter to switch to *vision* mode.

### Minimal example

```python
from g4f import ChatCompletion

answer = ChatCompletion.create(
    model="gpt-4", messages=[{"role": "user", "content": "Ping!"}]
)
print(answer)
```

---

## 2. `g4f.client` module

### `Client`

A one-stop object that contains nested service namespaces mirroring the official OpenAI Python SDK.

```python
client = g4f.client.Client(proxy="http://127.0.0.1:7890")

client.chat.completions.create(...)
client.images.generate(...)
```

Key attributes:

| Attribute | Type | Purpose |
| --------- | ---- | ------- |
| `chat.completions` | `Completions` | Synchronous chat endpoint. |
| `images` / `media` | `Images` | Image generation & variation helpers. |
| `models` | `ClientModels` | Convenience object for provider selection. |

### `AsyncClient`

Identical surface but all methods are `async`:

```python
async_client = g4f.client.AsyncClient()
answer = await async_client.chat.completions.create(...)
```

---

## 3. `g4f.models` module

### `Model`
Dataclass with fields:

```python
name: str           # "gpt-4o", "llama-3-8b", ...
base_provider: str  # human readable provider family
best_provider: ProviderType | IterListProvider
```

The file ships with **hundreds** of ready-to-use model constants (e.g. `g4f.models.gpt_4`, `llama_3_70b`, `dall_e_3`).  Retrieve them dynamically via:

```python
from g4f.models import ModelRegistry
print(ModelRegistry.all_models().keys())
```

### `ModelRegistry` – helper methods

* `get(name)` – resolve an alias or canonical name to a `Model` instance.
* `list_models_by_provider(provider_name)` – filter by provider (e.g. "Together").
* `validate_all_models()` – sanity-check that each registered model has a provider.

---

## 4. Error classes (`g4f.errors`)

| Error | Raised when |
| ----- | ----------- |
| `StreamNotSupportedError` | You requested `stream=True` but the selected provider lacks streaming support. |
| `NoMediaResponseError` | No image data was returned by the provider. |

All errors ultimately inherit from `Exception`.

---

## 5. Provider ecosystem (advanced)

Providers live in the `g4f.providers` & `g4f.Provider` packages.  Each provider class implements `create_function` and (optionally) `async_create_function`.  When adding a new provider follow the template in `providers/types.py`.

---

## 6. Typing aliases (`g4f.typing`)

Useful public aliases:

```python
Messages = list[dict[str, str]]
ImageType = Union[str, bytes, pathlib.Path, BinaryIO]
CreateResult = ChatCompletion | Iterator[ChatCompletionChunk]
```

---

## 7. CLI entry-point (`python -m g4f`)

`python -m g4f "Your prompt here" --model gpt-4o --stream` launches the minimal CLI tool defined in `g4f/__main__.py`.

---

## 8. Debug utilities

Set the `G4F_DEBUG` environment variable to enable verbose logs from the `g4f.debug` module.

---

### Notes

* **Stability** – Public APIs follow semantic-versioning rules (see `g4f.version.__version__`). Minor & patch releases will not introduce breaking changes.
* **Experimental modules** (`g4f.gui`, `g4f.local`) are *not* covered by this reference and may change at any time without notice.
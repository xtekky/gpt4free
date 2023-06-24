# DeepAI Wrapper
Written by [ading2210](https://github.com/ading2210/).

## Examples:
These functions are generators which yield strings containing the newly generated text.

### Completion:
```python
for chunk in deepai.Completion.create("Who are you?"):
    print(chunk, end="", flush=True)
print()
```

### Chat Completion:
Use the same format for the messages as you would for the [official OpenAI API](https://platform.openai.com/docs/guides/chat/introduction).
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
]
for chunk in deepai.ChatCompletion.create(messages):
    print(chunk, end="", flush=True)
print()
```
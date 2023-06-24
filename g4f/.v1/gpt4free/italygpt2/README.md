# Itagpt2(Rewrite)
Written by [sife-shuo](https://github.com/sife-shuo/).

## Description
Unlike gpt4free. italygpt in the pypi package, italygpt2 supports stream calls and has changed the request sending method to enable continuous and logical conversations.

The speed will increase when calling the conversation multiple times.

### Completion:
```python
account_data=italygpt2.Account.create()
for chunk in italygpt2.Completion.create(account_data=account_data,prompt="Who are you?"):
    print(chunk, end="", flush=True)
print()
```

### Chat
Like most chatgpt projects, format is supported.
Use the same format for the messages as you would for the [official OpenAI API](https://platform.openai.com/docs/guides/chat/introduction).
```python
messages = [
    {"role": "system", "content": ""},#...
    {"role": "user", "content": ""}#....
]
account_data=italygpt2.Account.create()
for chunk in italygpt2.Completion.create(account_data=account_data,prompt="Who are you?",message=messages):
    print(chunk, end="", flush=True)
print()
```
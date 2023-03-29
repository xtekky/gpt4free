working on it...

### `poe` (use like openai pypi package) - gpt-4

Import poe:

```python
import poe

# poe.Account.create
# poe.Completion.create
# poe.StreamCompletion.create
```

Create Token (3-6s)
```python
token = poe.Account.create(logging = True)
print('token', token)
```

Streaming Response
```python

for response in poe.StreamingCompletion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token):
    
    print(response.completion.choices[0].text, end="", flush=True)
```

Normal Response:
```python

response = poe.Completion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token)

print(response.completion.choices[0].text)    
```     






### `t3nsor` (use like openai pypi package)   

Import t3nsor:

```python
import t3nsor

# t3nsor.Completion.create
# t3nsor.StreamCompletion.create
```

Example Chatbot
```python
messages = []

while True:
    user = input('you: ')

    t3nsor_cmpl = t3nsor.Completion.create(
        prompt   = user,
        messages = messages
    )

    print('gpt:', t3nsor_cmpl.completion.choices[0].text)
    
    messages.extend([
        {'role': 'user', 'content': user }, 
        {'role': 'assistant', 'content': t3nsor_cmpl.completion.choices[0].text}
    ])
```

Streaming Response:

```python
for response in t3nsor.StreamCompletion.create(
    prompt   = 'write python code to reverse a string',
    messages = []):

    print(response.completion.choices[0].text)
```
### note: currently patched

### Example: `t3nsor` (use like openai pypi package) <a name="example-t3nsor"></a>

```python
# Import t3nsor
import t3nsor

# t3nsor.Completion.create
# t3nsor.StreamCompletion.create

[...]

```

#### Example Chatbot
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

#### Streaming Response:

```python
for response in t3nsor.StreamCompletion.create(
    prompt   = 'write python code to reverse a string',
    messages = []):

    print(response.completion.choices[0].text)
```

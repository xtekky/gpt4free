### Example: `you` (use like openai pypi package) <a name="example-you"></a>

```python
from you import Completion


completion = Completion()
response = completion.create(
    prompt='Hello, world!',
)
print(response)

chat_history = []
while True:
    prompt = input('You: ')

    completion.chat = chat_history
    response = completion.create(
        prompt=prompt,
    )
    print(response['response'])
    chat_history.append({"question": prompt, "answer": response["response"]})
```

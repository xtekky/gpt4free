ai.usesless.com

### Example: `usesless` <a name="example-usesless"></a>

### Token generation
<p>This will create account.json that contains email and token in json</p>

```python
from gpt4free import usesless


token = usesless.Account.create(logging=True)
print(token)
```

### Completion
<p>Insert token from account.json</p>

```python
import usesless

message_id = ""
token = <TOKENHERE> # usesless.Account.create(logging=True)
while True:
    prompt = input("Question: ")
    if prompt == "!stop":
        break

    req = usesless.Completion.create(prompt=prompt, parentMessageId=message_id, token=token)

    print(f"Answer: {req['text']}")
    message_id = req["id"]
```

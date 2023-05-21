ai.usesless.com

### Example: `usesless` <a name="example-usesless"></a>

### token generation
<p>this will create account.txt that contains mail and token</p>

```python
import usesless

usesless.Account.create(logging=True)
```

### completion
<p>insert token from account.txt</p>

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

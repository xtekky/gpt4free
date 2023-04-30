ai.usesless.com

to do:

- use random user agent in header
- make the code better I guess (?)

### Example: `usesless` <a name="example-usesless"></a>

```python
import usesless

message_id = ""
while True:
    prompt = input("Question: ")
    if prompt == "!stop":
        break

    req = usesless.Completion.create(prompt=prompt, parentMessageId=message_id)

    print(f"Answer: {req['text']}")
    message_id = req["id"]
```

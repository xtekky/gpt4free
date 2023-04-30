ai.usesless.com

to do:

- use random user agent in header
- make the code better I guess (?)

### Example: `usesless` <a name="example-usesless"></a>

```python
import usesless

question1 = "Who won the world series in 2020?"
req = usesless.Completion.create(prompt=question1)
answer = req["text"]
message_id = req["parentMessageId"]

question2 = "Where was it played?"
req2 = usesless.Completion.create(prompt=question2, parentMessageId=message_id)
answer2 = req2["text"]

print(answer)
print(answer2)
```

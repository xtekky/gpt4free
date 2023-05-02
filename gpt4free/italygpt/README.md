### Example: `italygpt`

```python
# create an instance
from gpt4free import italygpt
italygpt = italygpt.Completion()

# initialize api
italygpt.init()

# get an answer
italygpt.create(prompt="What is the meaning of life?")
print(italygpt.answer) # html formatted

# keep the old conversation
italygpt.create(prompt="Are you a human?", messages=italygpt.messages)
print(italygpt.answer)
```
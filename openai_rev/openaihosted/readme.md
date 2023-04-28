### Example: `openaihosted`) <a name="example-openaihosted"></a>

```python
# import library
from openai_rev import openaihosted

res = openaihosted.Completion.create(systemprompt="U are ChatGPT", text="What is 4+4",
                                     assistantprompt="U are a helpful assistant.").text
print(res)  ## Responds with the answer
```

### Example: `openaihosted`) <a name="example-openaihosted"></a>


```python
# import library
import openaihosted

res = openaihosted.Completion.create(systemprompt="U are ChatGPT", text="What is 4+4", assistantprompt="U are a helpful assistant.")['response']
print(res) ## Responds with the answer
```

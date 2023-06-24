### Example: `theb` (use like openai pypi package) <a name="example-theb"></a>

```python
# import library
from gpt4free import theb

# simple streaming completion

while True:
	x = input()
	for token in theb.Completion.create(x):
		print(token, end='', flush=True)
	print("")
```

### Example: `you` (use like openai pypi package) <a name="example-you"></a>

```python
import you

gpt = you.Completion()

# simple request with links and details
response = gpt.create(
    prompt       = "hello world",
    detailed     = True,
    includelinks = True
)

print(response)

# {
#     "response": "...",
#     "links": [...],
#     "extra": {
#         "youChatSerpResults": [...]
#     }
# }

while True:
    prompt = input("You: ")
    response = gpt.create(prompt)
    print("Bot:", response["response"])
```

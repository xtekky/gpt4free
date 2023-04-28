### Example: `forefront` (use like openai pypi package) <a name="example-forefront"></a>

```python
import forefront

# create an account
token = forefront.Account.create(logging=False)
print(token)

# get a response
for response in forefront.StreamingCompletion.create(token = token,
    prompt = 'hello world', model='gpt-4'):
    
    print(response.completion.choices[0].text, end = '')
print("")
```

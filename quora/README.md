### Example: `quora (poe)` (use like openai pypi package) - GPT-4 <a name="example-poe"></a>

```python
# quora model names: (use left key as argument)
models = {
    'sage'   : 'capybara',
    'gpt-4'  : 'beaver',
    'claude-v1.2'         : 'a2_2',
    'claude-instant-v1.0' : 'a2',
    'gpt-3.5-turbo'       : 'chinchilla'
}
```

#### !! new: bot creation

```python
# import quora (poe) package
import quora

# create account
# make shure to set enable_bot_creation to True
token = quora.Account.create(logging = True, enable_bot_creation=True)

model = quora.Model.create(
    token = token,
    model = 'gpt-3.5-turbo', # or claude-instant-v1.0
    system_prompt = 'you are ChatGPT a large language model ...' 
)

print(model.name) # gptx....

# streaming response
for response in quora.StreamingCompletion.create(
    custom_model = model.name,
    prompt       ='hello world',
    token        = token):
    
    print(response.completion.choices[0].text)
```

#### Normal Response:
```python

response = quora.Completion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token)

print(response.completion.choices[0].text)    
```     


> ⚠  Warning !!!    
poe.com added security and can detect if you are making automated requests. You may get your account banned if you are using this api.  
The normal non-driver api is also currently not very stable     


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

### New: bot creation

```python
# import quora (poe) package
import quora

# create account
# make sure to set enable_bot_creation to True
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

### Normal Response:
```python

response = quora.Completion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token)

print(response.completion.choices[0].text)    
```     

### Update Use This For Poe
```python
from quora import Poe

# available models:  ['Sage', 'GPT-4', 'Claude+', 'Claude-instant', 'ChatGPT', 'Dragonfly', 'NeevaAI']

poe = Poe(model='ChatGPT', driver='firefox', cookie_path='cookie.json', driver_path='path_of_driver')
poe.chat('who won the football world cup most?')

# new bot creation
poe.create_bot('new_bot_name', prompt='You are new test bot', base_model='gpt-3.5-turbo')

```

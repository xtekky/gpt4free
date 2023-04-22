### Example: `ora` (use like openai pypi package) <a name="example-ora"></a>

### load model (new)

more gpt4 models in `/testing/ora_gpt4.py`

```python
# normal gpt-4: b8b12eaa-5d47-44d3-92a6-4d706f2bcacf
model = ora.CompletionModel.load(chatbot_id, 'gpt-4') # or gpt-3.5
```

#### create model / chatbot: 
```python
# inport ora
import ora

# create model
model = ora.CompletionModel.create(
    system_prompt = 'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible',
    description   = 'ChatGPT Openai Language Model',
    name          = 'gpt-3.5')

# init conversation (will give you a conversationId)
init = ora.Completion.create(
    model  = model,
    prompt = 'hello world')

print(init.completion.choices[0].text)

while True:
    # pass in conversationId to continue conversation
    
    prompt = input('>>> ')
    response = ora.Completion.create(
        model  = model,
        prompt = prompt,
        includeHistory = True, # remember history
        conversationId = init.id)
    
    print(response.completion.choices[0].text)
```
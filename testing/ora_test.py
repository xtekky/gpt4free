# inport ora
import ora

# create model
model = ora.CompletionModel.create(
    system_prompt = 'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible',
    description   = 'ChatGPT Openai Language Model',
    name          = 'gpt-3.5')

print(model.id)

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
        includeHistory = True,
        conversationId = init.id)
    
    print(response.completion.choices[0].text)
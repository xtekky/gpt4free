import quora

token = quora.Account.create(logging=True, enable_bot_creation=True)

model = quora.Model.create(
    token=token,
    model='gpt-3.5-turbo',  # or claude-instant-v1.0
    system_prompt='you are ChatGPT a large language model ...'
)

print(model.name)

for response in quora.StreamingCompletion.create(
        custom_model=model.name,
        prompt='hello world',
        token=token):
    print(response.completion.choices[0].text)

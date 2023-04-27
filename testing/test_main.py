from openai_rev import openai_rev, Provider, quora, forefront

# usage You
response = openai_rev.Completion.create(Provider.You, prompt='Write a poem on Lionel Messi')
print(response)

# usage Poe
token = quora.Account.create(logging=False)
response = openai_rev.Completion.create(
    Provider.Poe, prompt='Write a poem on Lionel Messi', token=token, model='ChatGPT'
)
print(response)

# usage forefront
token = forefront.Account.create(logging=False)
response = openai_rev.Completion.create(
    Provider.ForeFront, prompt='Write a poem on Lionel Messi', model='gpt-4', token=token
)
print(response)
print(f'END')

# usage theb
response = openai_rev.Completion.create(Provider.Theb, prompt='Write a poem on Lionel Messi')
print(response)

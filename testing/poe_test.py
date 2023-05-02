from time import sleep

from gpt4free import quora

token = quora.Account.create(proxy=None, logging=True)
print('token', token)

sleep(2)

for response in quora.StreamingCompletion.create(model='ChatGPT', prompt='hello world', token=token):
    print(response.text, flush=True)

quora.Account.delete(token)

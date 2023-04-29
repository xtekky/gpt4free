import theb

for token in theb.Completion.create('hello world'):
    print(token, end='', flush=True)

import vercelai

for token in vercelai.Completion.create('summarize the gnu gpl 1.0'):
    print(token, end='', flush=True)


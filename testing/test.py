import g4f

# Set with provider
stream = False
response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.Yqcloud, messages=[
                                     {"role": "user", "content": "hello"}], stream=stream)

if stream:
    for message in response:
        print(message)
else:
    print(response)
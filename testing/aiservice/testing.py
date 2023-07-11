from AiService import ChatCompletion

# Test 1
response = ChatCompletion.create(model="gpt-3.5-turbo",
                                 provider="AiService",
                                 stream=False,
                                 messages=[{'role': 'user', 'content': 'who are you?'}])

print(response)

# Test 2
response = ChatCompletion.create(model="gpt-3.5-turbo",
                                 provider="AiService",
                                 stream=False,
                                 messages=[{'role': 'user', 'content': 'what you can do?'}])

print(response)


# Test 3
response = ChatCompletion.create(model="gpt-3.5-turbo",
                                 provider="AiService",
                                 stream=False,
                                 messages=[
                                     {'role': 'user', 'content': 'now your name is Bob'},
                                     {'role': 'assistant', 'content': 'Hello Im Bob, you asistant'},
                                     {'role': 'user', 'content': 'what your name again?'},
                                 ])

print(response)
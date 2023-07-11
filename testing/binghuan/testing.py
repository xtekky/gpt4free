from BingHuan import ChatCompletion

# Test 1
response = ChatCompletion.create(model="gpt-3.5-turbo",
                                 provider="BingHuan",
                                 stream=False,
                                 messages=[{'role': 'user', 'content': 'who are you?'}])

print(response)

# Test 2
# this prompt will return emoji in end of response
response = ChatCompletion.create(model="gpt-3.5-turbo",
                                 provider="BingHuan",
                                 stream=False,
                                 messages=[{'role': 'user', 'content': 'what you can do?'}])

print(response)


# Test 3
response = ChatCompletion.create(model="gpt-4",
                                 provider="BingHuan",
                                 stream=False,
                                 messages=[
                                     {'role': 'user', 'content': 'now your name is Bob'},
                                     {'role': 'assistant', 'content': 'Hello Im Bob, you asistant'},
                                     {'role': 'user', 'content': 'what your name again?'},
                                 ])

print(response)
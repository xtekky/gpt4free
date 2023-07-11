from BingHuan import ChatCompletion
# text = "Hello, this is Bing. I can help you find information on the web, generate content such as poems, stories, code, essays, songs, celebrity parodies and more. I can also help you with rewriting, improving, or optimizing your content. Is there anything specific you would like me to help you with? 😊"
# print(text.encode('utf-8'))

# # Test 1
# response = ChatCompletion.create(model="gpt-3.5-turbo",
#                                  provider="BingHuan",
#                                  stream=False,
#                                  messages=[{'role': 'user', 'content': 'who are you?'}])

# print(response)

# Test 2
response = ChatCompletion.create(model="gpt-3.5-turbo",
                                 provider="BingHuan",
                                 stream=False,
                                 messages=[{'role': 'user', 'content': 'what you can do?'}])

print(response)


# Test 3
# response = ChatCompletion.create(model="gpt-4",
#                                  provider="BingHuan",
#                                  stream=False,
#                                  messages=[
#                                      {'role': 'user', 'content': 'now your name is Bob'},
#                                      {'role': 'assistant', 'content': 'Hello Im Bob, you asistant'},
#                                      {'role': 'user', 'content': 'what your name again?'},
#                                  ])

# print(response)
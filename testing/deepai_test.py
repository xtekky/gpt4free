from gpt4free import deepai

#single completion
for chunk in deepai.Completion.create("Write a list of possible vacation destinations:"):
    print(chunk, end="", flush=True)
print()

#chat completion
print("==============")
messages = [ #taken from the openai docs
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
]
for chunk in deepai.ChatCompletion.create(messages):
    print(chunk, end="", flush=True)
print()
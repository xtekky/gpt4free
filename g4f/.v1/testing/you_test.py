from gpt4free import you

# simple request with links and details
response = you.Completion.create(prompt="hello world", detailed=True, include_links=True)

print(response)

# {
#     "response": "...",
#     "links": [...],
#     "extra": {...},
#         "slots": {...}
#     }
# }

# chatbot

chat = []

while True:
    prompt = input("You: ")

    response = you.Completion.create(prompt=prompt, chat=chat)

    print("Bot:", response.text)

    chat.append({"question": prompt, "answer": response.text})

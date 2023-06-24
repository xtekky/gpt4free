import openaihosted

messages = [{"role": "system", "content": "You are a helpful assistant."}]
while True:
    question = input("Question: ")
    if question == "!stop":
        break

    messages.append({"role": "user", "content": question})
    request = openaihosted.Completion.create(messages=messages)

    response = request["responses"]
    messages.append({"role": "assistant", "content": response})
    print(f"Answer: {response}")

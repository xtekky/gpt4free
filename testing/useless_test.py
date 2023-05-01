from gpt4free import usesless

message_id = ""
while True:
    prompt = input("Question: ")
    if prompt == "!stop":
        break

    req = usesless.Completion.create(prompt=prompt, parentMessageId=message_id)

    print(f"Answer: {req['text']}")
    message_id = req["id"]

import gpt4free

message_id = ""
while True:
    prompt = input("Question: ")
    if prompt == "!stop":
        break

    req = gpt4free.Completion.create(provider=gpt4free.Provider.UseLess, prompt=prompt, parentMessageId=message_id)

    print(f"Answer: {req['text']}")
    message_id = req["id"]

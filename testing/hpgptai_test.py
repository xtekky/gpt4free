import hpgptai

#single completion
res = hpgptai.Completion.create("你是谁","127.0.0.1:7890")
print(res["reply"])


#chat completion
messages = [
    {
        "content": "你是谁",
        "html": "你是谁",
        "id": hpgptai.ChatCompletion.randomStr(),
        "role": "user",
        "who": "User: ",
    },
    {
        "content": "我是一位AI助手，专门为您提供各种服务和支持。我可以回答您的问题，帮助您解决问题，提供相关信息，并执行一些任务。请随时告诉我您需要什么帮助。",
        "html": "我是一位AI助手，专门为您提供各种服务和支持。我可以回答您的问题，帮助您解决问题，提供相关信息，并执行一些任务。请随时告诉我您需要什么帮助。",
        "id": hpgptai.ChatCompletion.randomStr(),
        "role": "assistant",
        "who": "AI: ",
    },
    {
        "content": "我上一句问的是什么？",
        "html": "我上一句问的是什么？",
        "id": hpgptai.ChatCompletion.randomStr(),
        "role": "user",
        "who": "User: ",
    },
]
res = hpgptai.ChatCompletion.create(messages,proxy="127.0.0.1:7890")
print(res["reply"])









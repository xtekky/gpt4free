from g4f import Provider

import g4f

Conversation = Provider.PI.Start_Conversation()

Chat_History = Provider.PI.GetChatHistory(Conversation)

response = g4f.ChatCompletion.create(
    model="pi", 
    provider=g4f.Provider.PI,
    messages=[
        {
            "role": "user",
            "content": 'Hello who are you?'
        }
    ], 
    stream=False,
    conversation=Conversation
)

for message in response:
    print(message, flush=True, end='')
    
Chat_Title = Provider.PI.GetConversationTitle(Conversation)

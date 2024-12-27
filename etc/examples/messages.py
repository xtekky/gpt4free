from g4f.client import Client

class ConversationHandler:
    def __init__(self, model="gpt-4"):
        self.client = Client()
        self.model = model
        self.conversation_history = []
        
    def add_user_message(self, content):
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
        
    def get_response(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history
        )
        assistant_message = {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        }
        self.conversation_history.append(assistant_message)
        return assistant_message["content"]

# Usage example
conversation = ConversationHandler()
conversation.add_user_message("Hello!")
print("Assistant:", conversation.get_response())

conversation.add_user_message("How are you?")
print("Assistant:", conversation.get_response())

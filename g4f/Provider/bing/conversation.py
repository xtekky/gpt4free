from aiohttp import ClientSession


class Conversation():
    def __init__(self, conversationId: str, clientId: str, conversationSignature: str) -> None:
        self.conversationId = conversationId
        self.clientId = clientId
        self.conversationSignature = conversationSignature

async def create_conversation(session: ClientSession, proxy: str = None) -> Conversation:
    url = 'https://www.bing.com/turing/conversation/create?bundleVersion=1.1199.4'
    async with session.get(url, proxy=proxy) as response:
        data = await response.json()

        conversationId = data.get('conversationId')
        clientId = data.get('clientId')
        conversationSignature = response.headers.get('X-Sydney-Encryptedconversationsignature')

        if not conversationId or not clientId or not conversationSignature:
            raise Exception('Failed to create conversation.')
        return Conversation(conversationId, clientId, conversationSignature)
        
async def list_conversations(session: ClientSession) -> list:
    url = "https://www.bing.com/turing/conversation/chats"
    async with session.get(url) as response:
        response = await response.json()
        return response["chats"]
        
async def delete_conversation(session: ClientSession, conversation: Conversation, proxy: str = None) -> list:
    url = "https://sydney.bing.com/sydney/DeleteSingleConversation"
    json = {
        "conversationId": conversation.conversationId,
        "conversationSignature": conversation.conversationSignature,
        "participant": {"id": conversation.clientId},
        "source": "cib",
        "optionsSets": ["autosave"]
    }
    try:
        async with session.post(url, json=json, proxy=proxy) as response:
            response = await response.json()
            return response["result"]["value"] == "Success"
    except:
        return False
from __future__ import annotations

import uuid
from aiohttp import ClientSession

class Conversation:
    """
    Represents a conversation with specific attributes.
    """
    def __init__(self, conversationId: str, clientId: str, conversationSignature: str) -> None:
        """
        Initialize a new conversation instance.

        Args:
        conversationId (str): Unique identifier for the conversation.
        clientId (str): Client identifier.
        conversationSignature (str): Signature for the conversation.
        """
        self.conversationId = conversationId
        self.clientId = clientId
        self.conversationSignature = conversationSignature

async def create_conversation(session: ClientSession, proxy: str = None) -> Conversation:
    """
    Create a new conversation asynchronously.

    Args:
    session (ClientSession): An instance of aiohttp's ClientSession.
    proxy (str, optional): Proxy URL. Defaults to None.

    Returns:
    Conversation: An instance representing the created conversation.
    """
    url = 'https://www.bing.com/search?toncp=0&FORM=hpcodx&q=Bing+AI&showconv=1&cc=en'
    async with session.get(url, proxy=proxy) as response:
        response.raise_for_status()
    headers = {
        "accept": "application/json",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-ms-client-request-id": str(uuid.uuid4()),
        "x-ms-useragent": "azsdk-js-api-client-factory/1.0.0-beta.1 core-rest-pipeline/1.12.3 OS/Windows",
        "referer": url,
        "Cookie": "; ".join(f"{c.key}={c.value}" for c in session.cookie_jar)
    }
    for k, v in headers.items():
        session.headers[k] = v
    url = 'https://www.bing.com/turing/conversation/create?bundleVersion=1.1579.2'
    async with session.get(url, headers=headers, proxy=proxy) as response:
        try:
            data = await response.json()
        except:
            raise RuntimeError(f"Response: {await response.text()}")

        conversationId = data.get('conversationId')
        clientId = data.get('clientId')
        conversationSignature = response.headers.get('X-Sydney-Encryptedconversationsignature')

        if not conversationId or not clientId or not conversationSignature:
            raise Exception('Failed to create conversation.')
        return Conversation(conversationId, clientId, conversationSignature)
        
async def list_conversations(session: ClientSession) -> list:
    """
    List all conversations asynchronously.

    Args:
    session (ClientSession): An instance of aiohttp's ClientSession.

    Returns:
    list: A list of conversations.
    """
    url = "https://www.bing.com/turing/conversation/chats"
    async with session.get(url) as response:
        response = await response.json()
        return response["chats"]
        
async def delete_conversation(session: ClientSession, conversation: Conversation, proxy: str = None) -> bool:
    """
    Delete a conversation asynchronously.

    Args:
    session (ClientSession): An instance of aiohttp's ClientSession.
    conversation (Conversation): The conversation to delete.
    proxy (str, optional): Proxy URL. Defaults to None.

    Returns:
    bool: True if deletion was successful, False otherwise.
    """
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
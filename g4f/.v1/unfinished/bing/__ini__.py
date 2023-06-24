# Import necessary libraries
import asyncio
from json import dumps, loads
from ssl import create_default_context

import websockets
from browser_cookie3 import edge
from certifi import where
from requests import get

# Set up SSL context
ssl_context = create_default_context()
ssl_context.load_verify_locations(where())


def format(msg: dict) -> str:
    """Format message as JSON string with delimiter."""
    return dumps(msg) + '\x1e'


def get_token():
    """Retrieve token from browser cookies."""
    cookies = {c.name: c.value for c in edge(domain_name='bing.com')}
    return cookies['_U']


class AsyncCompletion:
    async def create(
            prompt: str = 'hello world',
            optionSets: list = [
                'deepleo',
                'enable_debug_commands',
                'disable_emoji_spoken_text',
                'enablemm',
                'h3relaxedimg'
            ],
            token: str = get_token()):
        """Create a connection to Bing AI and send the prompt."""

        # Send create request
        create = get('https://edgeservices.bing.com/edgesvc/turing/conversation/create',
                     headers={
                         'host': 'edgeservices.bing.com',
                         'authority': 'edgeservices.bing.com',
                         'cookie': f'_U={token}',
                         'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69',
                     }
                     )

        # Extract conversation data
        conversationId = create.json()['conversationId']
        clientId = create.json()['clientId']
        conversationSignature = create.json()['conversationSignature']

        # Connect to WebSocket
        wss = await websockets.connect('wss://sydney.bing.com/sydney/ChatHub', max_size=None, ssl=ssl_context,
                                       extra_headers={
                                           # Add necessary headers
                                       }
                                       )

        # Send JSON protocol version
        await wss.send(format({'protocol': 'json', 'version': 1}))
        await wss.recv()

        # Define message structure
        struct = {
            # Add necessary message structure
        }

        # Send message
        await wss.send(format(struct))

        # Process responses
        base_string = ''
        final = False
        while not final:
            objects = str(await wss.recv()).split('\x1e')
            for obj in objects:
                if obj is None or obj == '':
                    continue

                response = loads(obj)
                if response.get('type') == 1 and response['arguments'][0].get('messages', ):
                    response_text = response['arguments'][0]['messages'][0]['adaptiveCards'][0]['body'][0].get(
                        'text')

                    yield (response_text.replace(base_string, ''))
                    base_string = response_text

                elif response.get('type') == 2:
                    final = True

        await wss.close()


async def run():
    """Run the async completion and print the result."""
    async for value in AsyncCompletion.create(
            prompt='summarize cinderella with each word beginning with a consecutive letter of the alphabet, a-z',
            optionSets=[
                "galileo",
            ]
    ):
        print(value, end='', flush=True)


asyncio.run(run())

import websockets
from json import dumps, loads

async def test():
    async with websockets.connect('wss://chatgpt.func.icu/conversation+ws') as wss:
        
        await wss.send(dumps(separators=(',', ':'), obj = {
            'content_type':'text',
            'engine':'chat-gpt',
            'parts':['hello world'],
            'options':{}
            }
        ))
        
        ended = None

        while not ended:
            try:
                response      = await wss.recv()
                json_response = loads(response)
                print(json_response)
                
                ended         = json_response.get('eof')
                
                if not ended:
                    print(json_response['content']['parts'][0])
                
            except websockets.ConnectionClosed:
                break


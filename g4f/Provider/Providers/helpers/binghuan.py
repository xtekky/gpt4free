# Original Code From : https://gitler.moe/g4f/gpt4free
# https://gitler.moe/g4f/gpt4free/src/branch/main/g4f/Provider/Providers/helpers/bing.py
import sys
import ssl
import uuid 
import json
import time
import random
import asyncio
import certifi
# import requests
from curl_cffi import requests
import websockets
import browser_cookie3

config = json.loads(sys.argv[1])

ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())



conversationstyles = {
    'gpt-4': [ #'precise'
        "nlu_direct_response_filter",
        "deepleo",
        "disable_emoji_spoken_text",
        "responsible_ai_policy_235",
        "enablemm",
        "h3precise",
        "rcsprtsalwlst",
        "dv3sugg",
        "autosave",
        "clgalileo",
        "gencontentv3"
    ],
    'balanced': [
        "nlu_direct_response_filter",
        "deepleo",
        "disable_emoji_spoken_text",
        "responsible_ai_policy_235",
        "enablemm",
        "harmonyv3",
        "rcsprtsalwlst",
        "dv3sugg",
        "autosave"
    ],
    'gpt-3.5-turbo': [ #'precise'
        "nlu_direct_response_filter",
        "deepleo",
        "disable_emoji_spoken_text",
        "responsible_ai_policy_235",
        "enablemm",
        "h3imaginative",
        "rcsprtsalwlst",
        "dv3sugg",
        "autosave",
        "gencontentv3"
    ]
}

def format(msg: dict) -> str:
    return json.dumps(msg) + '\x1e'

def get_token():
    return
    
    try:
        cookies = {c.name: c.value for c in browser_cookie3.edge(domain_name='bing.com')}
        return cookies['_U']
    except:
        print('Error: could not find bing _U cookie in edge browser.')
        exit(1)

class AsyncCompletion:
    async def create(
        prompt     : str = None,
        optionSets : list = None,
        token     : str = None): # No auth required anymore
        
        create = None
        for _ in range(5):
            try:
                create = requests.get('https://b.ai-huan.xyz/turing/conversation/create', 
                    headers = {
                        'host': 'b.ai-huan.xyz',
                        'accept-encoding': 'gzip, deflate, br',
                        'connection': 'keep-alive',
                        'authority': 'b.ai-huan.xyz',
                        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                        'accept-language': 'en-US,en;q=0.9',
                        'cache-control': 'max-age=0',
                        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
                        'sec-ch-ua-arch': '"x86"',
                        'sec-ch-ua-bitness': '"64"',
                        'sec-ch-ua-full-version': '"110.0.1587.69"',
                        'sec-ch-ua-full-version-list': '"Chromium";v="110.0.5481.192", "Not A(Brand";v="24.0.0.0", "Microsoft Edge";v="110.0.1587.69"',
                        'sec-ch-ua-mobile': '?0',
                        'sec-ch-ua-model': '""',
                        'sec-ch-ua-platform': '"Windows"',
                        'sec-ch-ua-platform-version': '"15.0.0"',
                        'sec-fetch-dest': 'document',
                        'sec-fetch-mode': 'navigate',
                        'sec-fetch-site': 'none',
                        'sec-fetch-user': '?1',
                        'upgrade-insecure-requests': '1',
                        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69',
                        'x-edge-shopping-flag': '1',
                        'x-forwarded-for': f'13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}'
                    }            
                )

                conversationId        = create.json()['conversationId']
                clientId              = create.json()['clientId']
                conversationSignature = create.json()['conversationSignature']

            except Exception as e:
                time.sleep(0.5)
                continue
        
        if create == None: raise Exception('Failed to create conversation.')

        wss: websockets.WebSocketClientProtocol or None = None

        wss = await websockets.connect('wss://sydney.vcanbb.chat/sydney/ChatHub', max_size = None, ssl = ssl_context,
            extra_headers = {
                'accept': 'application/json',
                'accept-language': 'en-US,en;q=0.9',
                'content-type': 'application/json',
                'sec-ch-ua': '"Not_A Brand";v="99", Microsoft Edge";v="110", "Chromium";v="110"',
                'sec-ch-ua-arch': '"x86"',
                'sec-ch-ua-bitness': '"64"',
                'sec-ch-ua-full-version': '"109.0.1518.78"',
                'sec-ch-ua-full-version-list': '"Chromium";v="110.0.5481.192", "Not A(Brand";v="24.0.0.0", "Microsoft Edge";v="110.0.1587.69"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-model': "",
                'sec-ch-ua-platform': '"Windows"',
                'sec-ch-ua-platform-version': '"15.0.0"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'x-ms-client-request-id': str(uuid.uuid4()),
                'x-ms-useragent': 'azsdk-js-api-client-factory/1.0.0-beta.1 core-rest-pipeline/1.10.0 OS/Win32',
                'Referer': 'https://b.ai-huan.xyz/search?q=Bing+AI&showconv=1&FORM=hpcodx',
                'Referrer-Policy': 'origin-when-cross-origin',
                'x-forwarded-for': f'13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}'
            }
        )

        await wss.send(format({'protocol': 'json', 'version': 1}))
        await wss.recv()

        struct = {
            'arguments': [
                {
                    'source': 'cib', 
                    'optionsSets': optionSets, 
                    'isStartOfSession': True, 
                    'message': {
                        'author': 'user', 
                        'inputMethod': 'Keyboard', 
                        'text': prompt, 
                        'messageType': 'Chat'
                    }, 
                    'conversationSignature': conversationSignature, 
                    'participant': {
                        'id': clientId
                    }, 
                    'conversationId': conversationId
                }
            ], 
            'invocationId': '0', 
            'target': 'chat', 
            'type': 4
        }
        
        await wss.send(format(struct))
        
        base_string = ''
        
        final = False
        while not final:
            objects = str(await wss.recv()).split('\x1e')
            for obj in objects:
                if obj is None or obj == '':
                    continue
                
                response = json.loads(obj)
                #print(response, flush=True, end='')
                if response.get('type') == 1 and response['arguments'][0].get('messages',):
                    response_text = response['arguments'][0]['messages'][0]['adaptiveCards'][0]['body'][0].get('text')
                    
                    yield (response_text.replace(base_string, ''))
                    base_string = response_text
        
                elif response.get('type') == 2:
                    final = True
        
        await wss.close()

# i thing bing realy donset understand multi message (based on prompt template)
def convert(messages):
    context = ""
    for message in messages:
        context += "[%s](#message)\n%s\n\n" % (message['role'],
                                               message['content'])
    return context

async def run(optionSets, messages):
    prompt = messages[-1]['content']
    if(len(messages) > 1):
        prompt = convert(messages)
    async for value in AsyncCompletion.create(prompt=prompt, optionSets=optionSets):     
        try:
            print(value, flush=True, end='')
        except UnicodeEncodeError as e:
            # emoji encoding problem
            print(value.encode('utf-8'), flush=True, end='')

optionSet = conversationstyles[config['model']]
asyncio.run(run(optionSet, config['messages']))
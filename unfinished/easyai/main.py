from requests import get
from os       import urandom
from json     import loads

sessionId = urandom(10).hex()

headers = {
    'Accept': 'text/event-stream',
    'Accept-Language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Referer': 'http://easy-ai.ink/chat',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'token': 'null',
}

while True:
    prompt = input('you: ')
    
    params = {
        'message': prompt,
        'sessionId': sessionId
    }

    for chunk in  get('http://easy-ai.ink/easyapi/v1/chat/completions', params = params,
        headers = headers, verify = False, stream = True).iter_lines():
        
        if b'content' in chunk:
            data = loads(chunk.decode('utf-8').split('data:')[1])
            print(data['content'], end='')
            
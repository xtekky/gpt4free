import requests

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
    params = {
        'message': 'what is my name',
        'sessionId': '2eacb8ad826056587598',
    }

    for chunk in  requests.get('http://easy-ai.ink/easyapi/v1/chat/completions', params=params,
        headers=headers,  verify=False, stream=True).iter_lines():
        
        if b'data:' in chunk:
            print(chunk)
        
        print(chunk)
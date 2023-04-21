import requests

headers = {
    'authority': 'www.sqlchat.ai',
    'accept': '*/*',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'content-type': 'text/plain;charset=UTF-8',
    'origin': 'https://www.sqlchat.ai',
    'referer': 'https://www.sqlchat.ai/',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
}

data = {
    'messages':[
        {'role':'system','content':''},
        {'role':'user','content':'hello world'},
    ],
    'openAIApiConfig':{
        'key':'',
        'endpoint':''
    }
}

response = requests.post('https://www.sqlchat.ai/api/chat', headers=headers, json=data, stream=True)
for message in response.iter_content(chunk_size=1024):
    print(message)
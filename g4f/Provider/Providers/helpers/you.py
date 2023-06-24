import sys
import json
import urllib.parse

from curl_cffi import requests

config = json.loads(sys.argv[1])
messages = config['messages']
prompt = ''


def transform(messages: list) -> list:
    result = []
    i = 0

    while i < len(messages):
        if messages[i]['role'] == 'user':
            question = messages[i]['content']
            i += 1

            if i < len(messages) and messages[i]['role'] == 'assistant':
                answer = messages[i]['content']
                i += 1
            else:
                answer = ''

            result.append({'question': question, 'answer': answer})

        elif messages[i]['role'] == 'assistant':
            result.append({'question': '', 'answer': messages[i]['content']})
            i += 1

        elif messages[i]['role'] == 'system':
            result.append({'question': messages[i]['content'], 'answer': ''})
            i += 1
            
    return result

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Sec-Fetch-Site': 'same-origin',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Sec-Fetch-Mode': 'navigate',
    'Host': 'you.com',
    'Origin': 'https://you.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15',
    'Referer': 'https://you.com/api/streamingSearch?q=nice&safeSearch=Moderate&onShoppingPage=false&mkt=&responseFilter=WebPages,Translations,TimeZone,Computation,RelatedSearches&domain=youchat&queryTraceId=7a6671f8-5881-404d-8ea3-c3f8301f85ba&chat=%5B%7B%22question%22%3A%22hi%22%2C%22answer%22%3A%22Hello!%20How%20can%20I%20assist%20you%20today%3F%22%7D%5D&chatId=7a6671f8-5881-404d-8ea3-c3f8301f85ba&__cf_chl_tk=ex2bw6vn5vbLsUm8J5rDYUC0Bjzc1XZqka6vUl6765A-1684108495-0-gaNycGzNDtA',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'document',
    'Priority': 'u=0, i',
}

if messages[-1]['role'] == 'user':
    prompt = messages[-1]['content']
    messages = messages[:-1]

params = urllib.parse.urlencode({
    'q': prompt,
    'domain': 'youchat',
    'chat': transform(messages)
})

def output(chunk):
    if b'"youChatToken"' in chunk:
        chunk_json = json.loads(chunk.decode().split('data: ')[1])

        print(chunk_json['youChatToken'], flush=True, end = '')

while True:
    try:
        response = requests.get(f'https://you.com/api/streamingSearch?{params}',
                        headers=headers, content_callback=output, impersonate='safari15_5')
        
        exit(0)
    
    except Exception as e:
        print('an error occured, retrying... |', e, flush=True)
        continue
import requests

token = requests.get('https://play.vercel.ai/openai.jpeg', headers={
    'authority': 'play.vercel.ai',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'referer': 'https://play.vercel.ai/',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}).text + '.'

headers = {
    'authority': 'play.vercel.ai',
    'custom-encoding': token,
    'origin': 'https://play.vercel.ai',
    'referer': 'https://play.vercel.ai/',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
}

for chunk in requests.post('https://play.vercel.ai/api/generate', headers=headers, stream=True, json={
    'prompt': 'hi',
    'model': 'openai:gpt-3.5-turbo',
    'temperature': 0.7,
    'maxTokens': 200,
    'topK': 1,
    'topP': 1,
    'frequencyPenalty': 1,
    'presencePenalty': 1,
    'stopSequences': []}).iter_lines():
    print(chunk)

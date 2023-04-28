import requests

class Completion:
    def create(prompt: str,
        model: str = 'openai:gpt-3.5-turbo',
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 1,
        top_k: int = 1,
        frequency_penalty: float = 1,
        presence_penalty: float = 1,
        stopSequences: list = []):
        
        token = requests.get('https://play.vercel.ai/openai.jpeg', headers={
            'authority': 'play.vercel.ai',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'referer': 'https://play.vercel.ai/',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}).text.replace('=','')
        
        print(token)

        headers = {
            'authority': 'play.vercel.ai',
            'custom-encoding': token,
            'origin': 'https://play.vercel.ai',
            'referer': 'https://play.vercel.ai/',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
        }

        for chunk in requests.post('https://play.vercel.ai/api/generate', headers=headers, stream=True, json={
                'prompt': prompt,
                'model': model,
                'temperature': temperature,
                'maxTokens': max_tokens,
                'topK': top_p,
                'topP': top_k,
                'frequencyPenalty': frequency_penalty,
                'presencePenalty': presence_penalty,
                'stopSequences': stopSequences}).iter_lines():
            
            yield (chunk)    
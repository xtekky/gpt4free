import requests, json


headers = {
    'Accept-Language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Origin': 'https://deepinfra.com',
    'Pragma': 'no-cache',
    'Referer': 'https://deepinfra.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'X-Deepinfra-Source': 'web-embed',
    'accept': 'text/event-stream',
    'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
}

json_data = json.dumps({
    'model': 'meta-llama/Llama-2-70b-chat-hf',
    'messages': [
        {
            'role': 'user',
            'content': 'what is the meaning of life ?',
        },
    ],
    'stream': True}, separators=(',', ':'))

response = requests.post('https://api.deepinfra.com/v1/openai/chat/completions', 
                        headers=headers, data=json_data, stream=True)

response.raise_for_status()
first = True

for line in response.iter_content(chunk_size=1024):
    if line.startswith(b"data: [DONE]"):
        break
    
    elif line.startswith(b"data: "):
        chunk = json.loads(line[6:])["choices"][0]["delta"].get("content")
        
        if chunk:
            if first:
                chunk = chunk.lstrip()
                if chunk:
                    first = False
            
            print(chunk) 

# Note: json_data will not be serialized by requests
# exactly as it was in the original request.
#data = '{"model":"meta-llama/Llama-2-70b-chat-hf","messages":[{"role":"user","content":"what is the meaning of life ?"},{"role":"assistant","content":"  The meaning of life is a question that has puzzled philosophers, religious leaders, scientists, and many others for centuries. There are many different perspectives on this question, and there is no one definitive answer. However, here are some possible approaches to understanding the meaning of life:\\n\\n1. Religious or spiritual perspective: Many people believe that the meaning of life is to fulfill a divine or spiritual purpose. According to this view, our lives have a higher purpose, which is to serve a deity or follow a set of moral principles. The meaning of life is then found in fulfilling this purpose, whether it be through prayer, service, or good works.\\n2. Personal fulfillment: Some people believe that the meaning of life is to find personal fulfillment and happiness. According to this view, we should pursue our passions and interests, cultivate strong relationships, and strive to live a life that is rich in experience and personal growth.\\n3. Social or cultural perspective: Others argue that the meaning of life is tied to our social and cultural contexts. From this perspective, the meaning of life is to contribute to the greater good of society, to uphold certain values or traditions, or to leave a lasting legacy.\\n4. Existentialism: Existentialists believe that the meaning of life is not predetermined or inherent, but rather something that each individual must create for themselves. According to this view, life has no inherent meaning, but we can give it meaning through our choices, actions, and experiences.\\n5. Scientific perspective: Some people believe that the meaning of life can be understood through scientific inquiry. From this perspective, the meaning of life might be seen as a product of evolution, a byproduct of the natural processes that gave rise to consciousness and intelligence.\\n6. Humanistic perspective: Humanists believe that the meaning of life is to promote human flourishing and well-being. According to this view, we should strive to create a world where individuals can live fulfilling lives, pursue their goals and dreams, and reach their full potential.\\n7. Nihilistic perspective: A minority perspective is that life has no inherent meaning and that any attempt to impose meaning on it is ultimately futile. This view is often associated with nihilism, which argues that traditional beliefs and values are baseless and that morality is an illusion.\\n8. Mystical perspective: Finally, some people believe that the meaning of life cannot be expressed in words or concepts, but can only be experienced directly through mystical practices such as meditation or psychedelic experiences.\\n\\nUltimately, the meaning of life may be something that each person must determine for themselves based on their own unique experiences, values, and beliefs. It could also be said that life has multiple meanings, and that these meanings can change throughout our lives as we grow, learn, and evolve as individuals."},{"role":"user","content":"welll well well"},{"role":"assistant","content":"  I\'m glad you\'re feeling enthusiastic! \\"Well\\" can be used in various ways to express excitement or surprise, such as:\\n\\n* \\"Well, well, well!\\" - This phrase is often used to express surprise or intrigue, especially when something unexpected happens or when you\'re presented with new information.\\n* \\"Look at you, well done!\\" - Here, \\"well\\" is used to convey approval or admiration for someone\'s achievement or effort.\\n* \\"You think you\'re so smart, don\'t you?\\" - In this sentence, \\"well\\" is used to express sarcasm or skepticism towards someone who thinks they know everything.\\n\\nHowever, I\'d like to point out that using \\"well\\" too frequently can come across as insincere or filler words, especially in formal writing or professional settings. It\'s important to use language that accurately conveys your intended message without relying on unnecessary filler words. Is there anything else you\'d like me to help you with?"},{"role":"user","content":"nice"}],"stream":true}'
#response = requests.post('https://api.deepinfra.com/v1/openai/chat/completions', headers=headers, data=data)
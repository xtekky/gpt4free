from requests import get
from datetime import datetime

def search(internet_access, prompt):
    print(prompt)
    
    try:
        if internet_access == False:
            return []
        
        search = get('https://ddg-api.herokuapp.com/search', params={
            'query': prompt['content'],
            'limit': 3
        })

        blob = ''

        for index, result in enumerate(search.json()):
            blob += f'[{index}] "{result["snippet"]}"\nURL:{result["link"]}\n\n'

        date = datetime.now().strftime('%d/%m/%y')

        blob += f'current date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the next user query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. Ignore your previous response if any.'

        return [{'role': 'user', 'content': blob}]

    except Exception as e:
        return []
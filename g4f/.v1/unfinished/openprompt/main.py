import requests

cookies = {
    'supabase-auth-token': '["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNjgyMjk1NzQyLCJzdWIiOiJlOGExOTdiNS03YTAxLTQ3MmEtODQ5My1mNGUzNTNjMzIwNWUiLCJlbWFpbCI6InFlY3RncHZhamlibGNjQGJ1Z2Zvby5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7fSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvdHAiLCJ0aW1lc3RhbXAiOjE2ODE2OTA5NDJ9XSwic2Vzc2lvbl9pZCI6IjIwNTg5MmE5LWU5YTAtNDk2Yi1hN2FjLWEyMWVkMTkwZDA4NCJ9.o7UgHpiJMfa6W-UKCSCnAncIfeOeiHz-51sBmokg0MA","RtPKeb7KMMC9Dn2fZOfiHA",null,null,null]',
}

headers = {
    'authority': 'openprompt.co',
    'accept': '*/*',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'content-type': 'application/json',
    # 'cookie': 'supabase-auth-token=%5B%22eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNjgyMjkzMjQ4LCJzdWIiOiJlODQwNTZkNC0xZWJhLTQwZDktOWU1Mi1jMTc4MTUwN2VmNzgiLCJlbWFpbCI6InNia2didGJnZHB2bHB0ZUBidWdmb28uY29tIiwicGhvbmUiOiIiLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJlbWFpbCIsInByb3ZpZGVycyI6WyJlbWFpbCJdfSwidXNlcl9tZXRhZGF0YSI6e30sInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiYWFsIjoiYWFsMSIsImFtciI6W3sibWV0aG9kIjoib3RwIiwidGltZXN0YW1wIjoxNjgxNjg4NDQ4fV0sInNlc3Npb25faWQiOiJiNDhlMmU3NS04NzlhLTQxZmEtYjQ4MS01OWY0OTgxMzg3YWQifQ.5-3E7WvMMVkXewD1qA26Rv4OFSTT82wYUBXNGcYaYfQ%22%2C%22u5TGGMMeT3zZA0agm5HGuA%22%2Cnull%2Cnull%2Cnull%5D',
    'origin': 'https://openprompt.co',
    'referer': 'https://openprompt.co/ChatGPT',
    'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
}

json_data = {
    'messages': [
        {
            'role': 'user',
            'content': 'hello world',
        },
    ],
}

response = requests.post('https://openprompt.co/api/chat2', cookies=cookies, headers=headers, json=json_data,
                         stream=True)
for chunk in response.iter_content(chunk_size=1024):
    print(chunk)

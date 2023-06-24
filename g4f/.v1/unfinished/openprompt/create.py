from json import dumps
# from mail import MailClient
from re import findall

from requests import post, get

html = get('https://developermail.com/mail/')
print(html.cookies.get('mailboxId'))
email = findall(r'mailto:(.*)">', html.text)[0]

headers = {
    'apikey': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVzanNtdWZ1emRjcnJjZXVobnlqIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NzgyODYyMzYsImV4cCI6MTk5Mzg2MjIzNn0.2MQ9Lkh-gPqQwV08inIgqozfbYm5jdYWtf-rn-wfQ7U',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'x-client-info': '@supabase/auth-helpers-nextjs@0.5.6',
}

json_data = {
    'email': email,
    'password': 'T4xyt4Yn6WWQ4NC',
    'data': {},
    'gotrue_meta_security': {},
}

response = post('https://usjsmufuzdcrrceuhnyj.supabase.co/auth/v1/signup', headers=headers, json=json_data)
print(response.json())

# email_link = None
# while not email_link:
#     sleep(1)

#     mails = mailbox.getmails()
#     print(mails)


quit()

url = input("Enter the url: ")
response = get(url, allow_redirects=False)

# https://openprompt.co/#access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNjgyMjk0ODcxLCJzdWIiOiI4NWNkNTNiNC1lZTUwLTRiMDQtOGJhNS0wNTUyNjk4ODliZDIiLCJlbWFpbCI6ImNsc2J5emdqcGhiQGJ1Z2Zvby5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7fSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvdHAiLCJ0aW1lc3RhbXAiOjE2ODE2OTAwNzF9XSwic2Vzc2lvbl9pZCI6ImY4MTg1YTM5LTkxYzgtNGFmMy1iNzAxLTdhY2MwY2MwMGNlNSJ9.UvcTfpyIM1TdzM8ZV6UAPWfa0rgNq4AiqeD0INy6zV8&expires_in=604800&refresh_token=_Zp8uXIA2InTDKYgo8TCqA&token_type=bearer&type=signup

redirect = response.headers.get('location')
access_token = redirect.split('&')[0].split('=')[1]
refresh_token = redirect.split('&')[2].split('=')[1]

supabase_auth_token = dumps([access_token, refresh_token, None, None, None], separators=(',', ':'))
print(supabase_auth_token)

cookies = {
    'supabase-auth-token': supabase_auth_token
}

json_data = {
    'messages': [
        {
            'role': 'user',
            'content': 'how do I reverse a string in python?'
        }
    ]
}

response = post('https://openprompt.co/api/chat2', cookies=cookies, json=json_data, stream=True)
for chunk in response.iter_content(chunk_size=1024):
    print(chunk)

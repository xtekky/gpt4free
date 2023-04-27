# import writesonic
import writesonic

# create account (3-4s)
account = writesonic.Account.create(logging=True)

# with loging: 
# 2023-04-06 21:50:25 INFO __main__ -> register success : '{"id":"51aa0809-3053-44f7-922a...' (2s)
# 2023-04-06 21:50:25 INFO __main__ -> id : '51aa0809-3053-44f7-922a-2b85d8d07edf'
# 2023-04-06 21:50:25 INFO __main__ -> token : 'eyJhbGciOiJIUzI1NiIsInR5cCI6Ik...'
# 2023-04-06 21:50:28 INFO __main__ -> got key : '194158c4-d249-4be0-82c6-5049e869533c' (2s)

# simple completion
response = writesonic.Completion.create(
    api_key=account.key,
    prompt='hello world'
)

print(response.completion.choices[0].text)  # Hello! How may I assist you today?

# conversation

response = writesonic.Completion.create(
    api_key=account.key,
    prompt='what is my name ?',
    enable_memory=True,
    history_data=[
        {
            'is_sent': True,
            'message': 'my name is Tekky'
        },
        {
            'is_sent': False,
            'message': 'hello Tekky'
        }
    ]
)

print(response.completion.choices[0].text)  # Your name is Tekky.

# enable internet

response = writesonic.Completion.create(
    api_key=account.key,
    prompt='who won the quatar world cup ?',
    enable_google_results=True
)

print(response.completion.choices[0].text)  # Argentina won the 2022 FIFA World Cup tournament held in Qatar ...

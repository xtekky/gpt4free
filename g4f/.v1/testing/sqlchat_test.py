import sqlchat

for response in sqlchat.StreamCompletion.create(prompt='write python code to reverse a string', messages=[]):
    print(response.completion.choices[0].text, end='')

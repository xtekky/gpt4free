#!/usr/bin/python3

from gpt4free import forefront


# create an account
account_data = forefront.Account.create(logging=True)

# get a response
for response in forefront.StreamingCompletion.create(
    account_data=account_data,
    prompt='hello world',
    model='gpt-4'
):
    print('--------')
    print(response.choices[0].text, end='')
print("end")

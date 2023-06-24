import forefront
token = forefront.Account.create()
response = forefront.Completion.create(token=token, prompt='Hello!')
print(response)
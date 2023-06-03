# Fix by @enganese
# Import Account class from __init__.py file
from gpt4free import usesless

# Create Account and enable logging to see all the log messages (it's very interesting, try it!)
# New account credentials will be automatically saved in account.json file in such template: {"email": "username@1secmail.com", "token": "token here"}
token = usesless.Account.create(logging=True)

# Print the new token
print(token)

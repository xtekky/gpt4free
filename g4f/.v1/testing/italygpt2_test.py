from gpt4free import italygpt2
account_data=italygpt2.Account.create()
for chunk in italygpt2.Completion.create(account_data=account_data,prompt="Who are you?"):
    print(chunk, end="", flush=True)
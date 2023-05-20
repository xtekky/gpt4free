import forefront
import italygpt2
import quora
import theb
import usesless
import you

question="Who are you?"

#forefont
print("ForeFont:", end='')
account_data = forefront.Account.create(logging=False)
for response in forefront.StreamingCompletion.create(
    account_data=account_data,
    prompt=question,
    model='gpt-4'
):
    print(response.choices[0].text, end='')
print("")

#italygpt2
print("ItalyGPT:", end='')
account_data=italygpt2.Account.create()
for chunk in italygpt2.Completion.create(account_data=account_data,prompt=question):
    print(chunk, end="", flush=True)
print("")

#quora
print("Quora:", end='')
token = quora.Account.create(logging=True, enable_bot_creation=True)
model = quora.Model.create(
    token=token,
    model='gpt-3.5-turbo',  # or claude-instant-v1.0
)
for response in quora.StreamingCompletion.create(
        custom_model=model.name,
        prompt=question,
        token=token):
    print(response.completion.choices[0].text,end='')
print("")

#theb
print("Theb:", end='')
for token in theb.Completion.create(question):
	print(token, end='', flush=True)
print("")

#usesless
req = usesless.Completion.create(prompt=question, parentMessageId="")
print(f"Usesless: {req['text']}")
message_id = req["id"]

#you
response = you.Completion.create(prompt=question)
print("You:", response.text)
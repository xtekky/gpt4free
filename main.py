from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="blackboxai",
    messages=[{"role": "user", "content": "Hello"}],
    # Add any other necessary parameters
)
print(response.choices[0].message.content)

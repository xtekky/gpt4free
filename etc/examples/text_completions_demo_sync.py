from g4f.client import Client

client = Client()

response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "how does a court case get to the Supreme Court?"},
    ],
)

if not response.choices or response.choices[0].message is None:
    raise ValueError("LLM returned empty or filtered response")
print(response.choices[0].message.content)

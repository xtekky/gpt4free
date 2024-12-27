import g4f
import requests

from g4f.client import Client
from g4f.Provider.Blackbox import Blackbox

client = Client(
    provider=Blackbox
)

image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/images/cat.jpeg", stream=True).content
image = open("docs/images/cat.jpeg", "rb")

response = client.chat.completions.create(
    model=g4f.models.default,
    messages=[
        {"role": "user", "content": "What are on this image?"}
    ],
    image=image
)

print(response.choices[0].message.content)

import g4f
import requests

from g4f.client import Client
from g4f.Provider.Blackbox import Blackbox

client = Client(
    provider=Blackbox
)

# Processing remote image
remote_image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/images/cat.jpeg", stream=True).content
response_remote = client.chat.completions.create(
    model=g4f.models.default,
    messages=[
        {"role": "user", "content": "What are on this image?"}
    ],
    image=remote_image
)
print("Response for remote image:")
print(response_remote.choices[0].message.content)

print("\n" + "-"*50 + "\n")  # Separator

# Processing local image
local_image = open("docs/images/cat.jpeg", "rb")
response_local = client.chat.completions.create(
    model=g4f.models.default,
    messages=[
        {"role": "user", "content": "What are on this image?"}
    ],
    image=local_image
)
print("Response for local image:")
print(response_local.choices[0].message.content)
local_image.close()  # Close file after use

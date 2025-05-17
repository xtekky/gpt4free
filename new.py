import json
import uuid
import g4f.debug
import requests
from g4f.client import Client

def upload_and_process(files_or_urls, bucket_id=None):
    if bucket_id is None:
        bucket_id = str(uuid.uuid4())

    if isinstance(files_or_urls, list):  # URLs
        files = {'files': ('downloads.json', json.dumps(
            files_or_urls), 'application/json')}
    elif isinstance(files_or_urls, dict):  # Files
        files = files_or_urls
    else:
        raise ValueError(
            "files_or_urls must be a list of URLs or a dictionary of files")

    upload_response = requests.post(
        f'http://localhost:8080/backend-api/v2/files/{bucket_id}', files=files)

    if upload_response.status_code == 200:
        upload_data = upload_response.json()
        print(f"Upload successful. Bucket ID: {upload_data['bucket_id']}")
    else:
        print(
            f"Upload failed: {upload_response.status_code} - {upload_response.text}")

    response = requests.get(
        f'http://localhost:8080/backend-api/v2/files/{bucket_id}', stream=True, headers={'Accept': 'text/event-stream'})
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data:'):
                try:
                    data = json.loads(line[5:])  # remove data: prefix
                    if "action" in data:
                        print(f"SSE Event: {data}")
                    elif "error" in data:
                        print(f"Error: {data['error']['message']}")
                    else:
                        # Assuming it's file content
                        print(f"File data received: {data}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            else:
                print(f"Unhandled SSE event: {line}")
    response.close()
    return bucket_id


# Example with URLs

# Enable debug mode
g4f.debug.logging = True

client = Client()

# Upload example file
files = {'files': ('demo.docx', open('demo.docx', 'rb'))}
bucket_id = upload_and_process(files)

# Send request with file:
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Discribe this file."},
        {"bucket_id": bucket_id}
    ]}],
)
print(response.choices[0].message.content)
exit()
# import asyncio
# from g4f.client import AsyncClient
# import g4f.Provider

# async def main():
#     client = AsyncClient(provider=g4f.Provider.MarkItDown)

#     # Transcribe a audio file
#     with open("audio.wav", "rb") as audio_file:
#         response = await client.chat.completions.create("", media=[audio_file])
#         print(response.choices[0].message.content)

# if __name__ == "__main__":
#     asyncio.run(main())

#exit()
import requests

# Open the audio file in binary mode
with open('demo.docx', 'rb') as audio_file:
    # Make the POST request
    response = requests.post('http://localhost:8080/api/markitdown', files={'file': audio_file})

    # Check the response and print the transcription
    if response.status_code == 200:
        data = response.json()
        print(data['text'])
    else:
        print(f"Error: {response.status_code}, {response.text}")
exit()

# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:8080/v1", api_key="secret")

# with open("audio.wav", "rb") as file:
#     transcript = client.audio.transcriptions.create(
#         model="",
#         extra_body={"provider": "MarkItDown"},
#         file=file
#     )
# print(transcript.text)

exit()
import asyncio
import time
from g4f import AsyncClient
from g4f.Provider import PollinationsAI
async def test():
    client = AsyncClient()
    response = client.chat.completions.create("guten tag", stream=True, provider=HarProvider)
    async for chunk in response:
        if chunk.choices[0].finish_reason == "stop":
            break
        print(chunk.choices[0].delta.content, end="", flush=True)
    print()

asyncio.run(test())
time.sleep(1)
exit()

client = Client(provider=PollinationsAI)
response = client.media.generate("Hello", model="hypnosis-tracy")
response.data[0].save("hypnosis.mp3")

client = Client(provider=Gemini)
response = client.media.generate("Hello", model="gemini-audio")
response.data[0].save("gemini.ogx")

client = Client(provider=EdgeTTS)
response = client.media.generate("Hello", audio={"locale": "en-US"})
response.data[0].save("edge-tts.mp3")

exit()
import requests
import uuid
import json

def upload_and_process(files_or_urls, bucket_id=None):
    if bucket_id is None:
        bucket_id = str(uuid.uuid4())
    
    if isinstance(files_or_urls, list): #URLs
        files = {'files': ('downloads.json', json.dumps(files_or_urls), 'application/json')}
    elif isinstance(files_or_urls, dict): #Files
        files = files_or_urls
    else:
        raise ValueError("files_or_urls must be a list of URLs or a dictionary of files")

    upload_response = requests.post(f'http://localhost:8080/v1/files/{bucket_id}', files=files)

    if upload_response.status_code == 200:
        upload_data = upload_response.json()
        print(f"Upload successful. Bucket ID: {upload_data['bucket_id']}")
    else:
        print(f"Upload failed: {upload_response.status_code} - {upload_response.text}")

    response = requests.get(f'http://localhost:8080/v1/files/{bucket_id}', stream=True, headers={'Accept': 'text/event-stream'})
    for line in response.iter_lines():
      if line:
          line = line.decode('utf-8')
          if line.startswith('data:'):
              try:
                  data = json.loads(line[5:]) #remove data: prefix
                  if "action" in data:
                      print(f"SSE Event: {data}")
                  elif "error" in data:
                      print(f"Error: {data['error']['message']}")
                  else:
                      print(f"File data received: {data}") #Assuming it's file content
              except json.JSONDecodeError as e:
                  print(f"Error decoding JSON: {e}")
          else:
              print(f"Unhandled SSE event: {line}")
    response.close()
    return bucket_id

# Example with URLs
#Example with files
#files = {'files': open('document.pdf', 'rb'), 'files': open('data.json', 'rb')}
#bucket_id = upload_and_process(files)
import asyncio
from g4f.client import Client

import g4f.debug
g4f.debug.logging = True

client = Client()


files = {'files': ('demo.docx', open('demo.docx', 'rb'))}
bucket_id = upload_and_process(files)


response = client.chat.completions.create(
    [{"role": "user", "content": [
        {"type": "text", "text": "Discribe this file."},
        {"bucket_id": bucket_id}
    ]}],
    "o1",
)
print(response.choices[0].message.content)


### G4F - Media Documentation

This document outlines how to use the G4F (Generative Framework) library to generate and process various media types, including audio, images, and videos.

---

### 1. **Audio Generation and Transcription**

G4F supports audio generation through providers like PollinationsAI and audio transcription using providers like Microsoft_Phi_4.

#### **Generate Audio with PollinationsAI:**

```python
import asyncio
from g4f.client import AsyncClient
import g4f.Provider

async def main():
    client = AsyncClient(provider=g4f.Provider.PollinationsAI)

    response = await client.chat.completions.create(
        model="openai-audio",
        messages=[{"role": "user", "content": "Say good day to the world"}],
        audio={"voice": "alloy", "format": "mp3"},
    )
    response.choices[0].message.save("alloy.mp3")

asyncio.run(main())
```

#### **More examples for Generate Audio:**

```python
from g4f.client import Client

from g4f.Provider import gTTS, EdgeTTS, Gemini, PollinationsAI

client = Client(provider=PollinationsAI)
response = client.media.generate("Hello", audio={"voice": "alloy", "format": "mp3"})
response.data[0].save("openai.mp3")

client = Client(provider=PollinationsAI)
response = client.media.generate("Hello", model="hypnosis-tracy")
response.data[0].save("hypnosis.mp3")

client = Client(provider=Gemini)
response = client.media.generate("Hello", model="gemini-audio")
response.data[0].save("gemini.ogx")

client = Client(provider=EdgeTTS)
response = client.media.generate("Hello", audio={"language": "en"})
response.data[0].save("edge-tts.mp3")

# The EdgeTTS provider also support the audio parameters `rate`, `volume` and `pitch`

client = Client(provider=gTTS)
response = client.media.generate("Hello", audio={"language": "en-US"})
response.data[0].save("google-tts.mp3")

# The gTTS provider also support the audio parameters `tld` and `slow`
```

#### **Transcribe an Audio File:**

Some providers in G4F support audio inputs in chat completions, allowing you to transcribe audio files by instructing the model accordingly. This example demonstrates how to use the `AsyncClient` to transcribe an audio file asynchronously:

```python
import asyncio
from g4f.client import AsyncClient
import g4f.Provider

async def main():
    client = AsyncClient(provider=g4f.Provider.Microsoft_Phi_4)

    with open("audio.wav", "rb") as audio_file:
        response = await client.chat.completions.create(
            messages="Transcribe this audio",
            media=[[audio_file, "audio.wav"]],
            modalities=["text"],
        )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Explanation
- **Client Initialization**: An `AsyncClient` instance is created with a provider that supports audio inputs, such as `PollinationsAI` or `Microsoft_Phi_4`.
- **File Handling**: The audio file (`audio.wav`) is opened in binary read mode (`"rb"`) using a context manager (`with` statement) to ensure proper file closure after use.
- **API Call**: The `chat.completions.create` method is called with:
  - `messages`: Containing a user message instructing the model to transcribe the audio.
  - `media`: A list of lists, where each inner list contains the file object and its name (`[[audio_file, "audio.wav"]]`).
  - `modalities=["text"]`: Specifies that the output should be text (the transcription).
- **Response**: The transcription is extracted from `response.choices[0].message.content` and printed.

#### Notes
- **Provider Support**: Ensure the chosen provider (e.g., `PollinationsAI` or `Microsoft_Phi_4`) supports audio inputs in chat completions. Not all providers may offer this functionality.
- **File Path**: Replace `"audio.wav"` with the path to your own audio file. The file format (e.g., WAV) should be compatible with the provider.
- **Model Selection**: If `g4f.models.default` does not support audio transcription, you may need to specify a model that does (consult the provider's documentation for supported models).

This example complements the guide by showcasing how to handle audio inputs asynchronously, expanding on the multimodal capabilities of the G4F AsyncClient API.

---

### 2. **Image Generation**

G4F can generate images from text prompts and provides options to retrieve images as URLs or base64-encoded strings.

#### **Generate an Image:**

```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="url",
    )

    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

#### **Base64 Response Format:**

```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="b64_json",
    )

    base64_text = response.data[0].b64_json
    print(base64_text)

asyncio.run(main())
```

#### **Image Parameters:**
- **`width`**: Defines the width of the generated image.
- **`height`**: Defines the height of the generated image.
- **`n`**: Specifies the number of images to generate.
- **`response_format`**: Specifies the format of the response:
  - `"url"`: Returns the URL of the image.
  - `"b64_json"`: Returns the image as a base64-encoded JSON string.
  - (Default): Saves the image locally and returns a local url.

#### **Example with Image Parameters:**

```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="url",
        width=512,
        height=512,
        n=2,
    )

    for image in response.data:
        print(f"Generated image URL: {image.url}")

asyncio.run(main())
```

---

### 3. **Creating Image Variations**

You can generate variations of an existing image using G4F.

#### **Create Image Variations:**

```python
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import OpenaiChat

async def main():
    client = AsyncClient(image_provider=OpenaiChat)

    response = await client.images.create_variation(
        prompt="a white siamese cat",
        image=open("docs/images/cat.jpg", "rb"),
        model="dall-e-3",
    )

    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

---

### 4. **Video Generation**

G4F supports video generation through providers like HuggingFaceMedia.

#### **Generate a Video:**

```python
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import HuggingFaceMedia

async def main():
    client = AsyncClient(
        provider=HuggingFaceMedia,
        api_key=os.getenv("HF_TOKEN") # Your API key here
    )

    video_models = client.models.get_video()
    print("Available Video Models:", video_models)

    result = await client.media.generate(
        model=video_models[0],
        prompt="G4F AI technology is the best in the world.",
        response_format="url",
    )

    print("Generated Video URL:", result.data[0].url)

asyncio.run(main())
```

#### **Video Parameters:**
- **`resolution`**: Specifies the resolution of the generated video. Options include:
  - `"480p"` (default)
  - `"720p"`
- **`aspect_ratio`**: Defines the width-to-height ratio (e.g., `"16:9"`).
- **`n`**: Specifies the number of videos to generate.
- **`response_format`**: Specifies the format of the response:
  - `"url"`: Returns the URL of the video.
  - `"b64_json"`: Returns the video as a base64-encoded JSON string.
  - (Default): Saves the video locally and returns a local url.

#### **Example with Video Parameters:**

```python
import os
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import HuggingFaceMedia

async def main():
    client = AsyncClient(
        provider=HuggingFaceMedia,
        api_key=os.getenv("HF_TOKEN")  # Your API key here
    )

    video_models = client.models.get_video()
    print("Available Video Models:", video_models)

    result = await client.media.generate(
        model=video_models[0],
        prompt="G4F AI technology is the best in the world.",
        resolution="720p",
        aspect_ratio="16:9",
        n=1,
        response_format="url",
    )

    print("Generated Video URL:", result.data[0].url)

asyncio.run(main())
```

---

**Key Points:**

- **Provider Selection**: Ensure the selected provider supports the desired media generation or processing task.
- **API Keys**: Some providers require API keys for authentication.
- **Response Formats**: Use `response_format` to control the output format (URL, base64, local file).
- **Parameter Usage**: Use parameters like `width`, `height`, `resolution`, `aspect_ratio`, and `n` to customize the generated media.

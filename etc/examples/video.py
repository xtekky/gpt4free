import g4f.Provider
from g4f.client import Client

client = Client(
    provider=g4f.Provider.HuggingFaceMedia,
    api_key="hf_***" # Your API key here
)

video_models = client.models.get_video()

print(video_models)

result = client.media.generate(
    model=video_models[0],
    prompt="G4F AI technology is the best in the world.",
    response_format="url"
)

print(result.data[0].url)
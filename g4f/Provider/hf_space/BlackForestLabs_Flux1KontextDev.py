from __future__ import annotations

import os
import uuid

from ...typing import AsyncResult, Messages, MediaListType
from ...providers.response import ImageResponse, JsonConversation, Reasoning
from ...requests import StreamSession, FormData, sse_stream
from ...tools.media import merge_media
from ...image import to_bytes, is_accepted_format
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_media_prompt
from .DeepseekAI_JanusPro7b import get_zerogpu_token
from .raise_for_status import raise_for_status

class BlackForestLabs_Flux1KontextDev(AsyncGeneratorProvider, ProviderModelMixin):
    label = "BlackForestLabs Flux-1-Kontext-Dev"
    url = "https://black-forest-labs-flux-1-kontext-dev.hf.space"
    space = "black-forest-labs/FLUX.1-Kontext-Dev"
    referer = f"{url}/?__theme=system"
    working = True

    default_model = "flux-kontext-dev"
    default_image_model = default_model
    image_models = [default_model]
    models = image_models

    @classmethod
    def run(cls, method: str, session: StreamSession, conversation: JsonConversation, data: list = None):
        headers = {
            # Different accept header based on GET or POST
            "accept": "application/json" if method == "post" else "text/event-stream",
            "content-type": "application/json",
            "x-zerogpu-token": conversation.zerogpu_token,
            "x-zerogpu-uuid": conversation.zerogpu_uuid,
            "referer": cls.referer,
        }
        # Filter out headers where value is None (e.g., token not yet set)
        filtered_headers = {k: v for k, v in headers.items() if v is not None}

        if method == "post":
            # POST request to enqueue the job
            return session.post(f"{cls.url}/gradio_api/queue/join?__theme=system", **{
                "headers": filtered_headers,
                "json": {
                    "data": data,
                    "event_data": None,
                    "fn_index": 2,
                    "trigger_id": 7,      # Using trigger_id=7 per your example fetch
                    "session_hash": conversation.session_hash
                }
            })

        # GET request to receive the event stream result
        return session.get(f"{cls.url}/gradio_api/queue/data?session_hash={conversation.session_hash}", **{
            "headers": filtered_headers,
        })

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        media: MediaListType = None,
        proxy: str = None,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28,
        seed: int = 0,
        randomize_seed: bool = True,
        cookies: dict = None,
        api_key: str = None,
        zerogpu_uuid: str = None,
        **kwargs
    ) -> AsyncResult:
        # Create a conversation/session data container holding tokens and session hash
        conversation = JsonConversation(
            zerogpu_token=api_key,
            zerogpu_uuid=zerogpu_uuid or uuid.uuid4().hex,
            session_hash=uuid.uuid4().hex,
        )
        async with StreamSession(impersonate="chrome", proxy=proxy) as session:
            media = list(merge_media(media, messages))
            if media:
                data = FormData()
                for i in range(len(media)):
                    if media[i][1] is None and isinstance(media[i][0], str):
                        media[i] = media[i][0], os.path.basename(media[i][0])
                    media[i] = (to_bytes(media[i][0]), media[i][1])
                for image, image_name in media:
                    data.add_field(f"files", image, filename=image_name)
                async with session.post(f"{cls.url}/gradio_api/upload", params={"upload_id": conversation.session_hash}, data=data) as response:
                    await raise_for_status(response)
                    image_files = await response.json()
                media = [{
                    "path": image_file,
                    "url": f"{cls.url}/gradio_api/file={image_file}",
                    "orig_name": media[i][1],
                    "size": len(media[i][0]),
                    "mime_type": is_accepted_format(media[i][0]),
                    "meta": {
                        "_type": "gradio.FileData"
                    }
                } for i, image_file in enumerate(image_files)]
            if not media:
                raise ValueError("No media files provided for image generation.")

            # Format the prompt from messages, e.g. extract text or media description
            prompt = format_media_prompt(messages, prompt)

            # Build the data payload sent to the API
            data = [
                media.pop(),
                prompt,
                seed,                 
                randomize_seed,
                guidance_scale,
                num_inference_steps,
            ]

            # Fetch token if it's missing (calls a helper function to obtain a token)
            if conversation.zerogpu_token is None:
                conversation.zerogpu_uuid, conversation.zerogpu_token = await get_zerogpu_token(
                    cls.space, session, conversation, cookies
                )

            # POST the prompt and data to start generation job in the queue
            async with cls.run("post", session, conversation, data) as response:
                await raise_for_status(response)
                result_json = await response.json()
                assert result_json.get("event_id")  # Ensure we got an event id back

            # GET the event stream to receive updates and results asynchronously
            async with cls.run("get", session, conversation) as event_response:
                await raise_for_status(event_response)
                async for chunk in sse_stream(event_response):
                    if chunk.get("msg") == "process_starts":
                        yield Reasoning(label="Processing started")
                    elif chunk.get("msg") == "progress":
                        progress_data = chunk.get("progress_data", [])
                        progress_data = progress_data[0] if progress_data else {}
                        yield Reasoning(label="Processing image", status=f"{progress_data.get('index', 0)}/{progress_data.get('length', 0)}")
                    elif chunk.get("msg") == "process_completed":
                        url = chunk.get("output", {}).get("data", [{}])[0].get("url")
                        yield ImageResponse(url, prompt)
                        yield Reasoning(label="Completed", status="")
                        break
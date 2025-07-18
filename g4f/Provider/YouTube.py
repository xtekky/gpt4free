from __future__ import annotations

import os

try:
    import yt_dlp
    has_yt_dlp = True
except ImportError:
    has_yt_dlp = False

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import AudioResponse, VideoResponse, YouTubeResponse
from ..image.copy_images import get_media_dir
from .helper import format_media_prompt

class YouTube(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://youtube.com"
    working = has_yt_dlp

    default_model = "search"
    models = ["mp3", "1080p", "720p", "480p", "search"]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        prompt = format_media_prompt(messages, prompt)
        results = [{
            "id": line.split("?v=")[-1].split("&")[0],
            "url": line
        } for line in prompt.splitlines()
            if line.startswith("https://www.youtube.com/watch?v=")]
        provider = YouTubeProvider()
        if not results:
            results = await provider.search(prompt, max_results=10)
        new_results = []
        for result in results:
            video_url = result['url']
            has_video = False
            for message in messages:
                if isinstance(message.get("content"), str):
                    if video_url in message["content"] and (model == "search" or model in message["content"]):
                        has_video = True
                        break
            if has_video:
                continue
            new_results.append(result)
        if model == "search":
            yield YouTubeResponse([result["id"] for result in new_results[:5]], True)
        else:
            if new_results:
                video_url = new_results[0]['url']
                path = await provider.download(video_url, model=model, output_dir=get_media_dir())
                if path.endswith('.mp3'):
                    yield AudioResponse(f"/media/{os.path.basename(path)}")
                else:
                    yield VideoResponse(f"/media/{os.path.basename(path)}", prompt)
                yield f"\n\n[{video_url}]({video_url})\n\n"

class YouTubeProvider:
    """
    Search and download YouTube videos.

    model: "mp3" for audio only, or "high-definition" for best video
    """

    def __init__(self):
        pass

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search YouTube for videos matching the query.

        Returns a list of dicts with keys: title, url, id, duration
        """
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'skip_download': True,
        }
        search_url = f"ytsearch{max_results}:{query}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        results = []
        for entry in info.get('entries', []):
            results.append({
                'title': entry.get('title'),
                'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                'id': entry.get('id'),
                'duration': entry.get('duration'),
            })
        return results

    async def download(self, video_url: str, model: str = "720p", output_dir: str = ".") -> str:
        """
        Download a YouTube video.

        :param video_url: The video URL or video id
        :param model: "mp3" for audio, "high-definition" for best video
        :param output_dir: Download location
        :return: The path to the downloaded file
        """
        ydl_opts = {
            'outtmpl': f"{output_dir}/%(title)s{'' if model == 'mp3' else (' ' + model)}.%(ext)s",
            'quiet': True,
        }
        if model == "mp3":
            # Audio only, best quality
            ydl_opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192'
                }]
            })
        elif model == "1080p":
            ydl_opts.update({
                'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                'merge_output_format': 'mp4',
            })
        elif model == "720p":
                    ydl_opts.update({
                'format': 'bestvideo[height=720]+bestaudio/best[height=720]',
                'merge_output_format': 'mp4',
            })
        elif model == "480p":
            ydl_opts.update({
                'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                'merge_output_format': 'mp4',
            })
        else:
            raise ValueError("model must be 'mp3' or 'high-definition'")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.download([video_url])
            return ydl.prepare_filename(ydl.extract_info(video_url, download=True)).replace('.webm', '.mp3' if model == "mp3" else '.webm')
            # You can get actual file path via ydl.prepare_filename
        # This is a simplified return - usually, you would parse the output or check the directory
        return output_dir

# Example usage (async function to test)

async def demo():
    provider = YouTubeProvider()
    results = await provider.search("Python programming", max_results=2)
    print("Search results:", results)
    if results:
        video_url = results[0]['url']
        path = await provider.download(video_url, model="mp3")
        print("Downloaded to:", path)

# To actually run demo()
# asyncio.run(demo())

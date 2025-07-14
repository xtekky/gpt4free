from __future__ import annotations

import os

try:
    import yt_dlp
    has_yt_dlp = True
except ImportError:
    has_yt_dlp = False

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import AudioResponse, VideoResponse
from ..image.copy_images import get_media_dir
from .helper import format_media_prompt

class YouTube(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://youtube.com"
    working = has_yt_dlp
    use_nodriver = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        prompt = format_media_prompt(messages, prompt)
        provider = YouTubeProvider()
        results = await provider.search(prompt, max_results=1)
        if results:
            video_url = results[0]['url']
            path = await provider.download(video_url, model="mp3", output_dir=get_media_dir())
            if path.endswith('.mp3'):
                yield AudioResponse(f"/media/{os.path.basename(path)}")
            else:
                yield VideoResponse(f"/media/{os.path.basename(path)}", prompt)

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

    async def download(self, video_url: str, model: str = "high-definition", output_dir: str = ".") -> str:
        """
        Download a YouTube video.

        :param video_url: The video URL or video id
        :param model: "mp3" for audio, "high-definition" for best video
        :param output_dir: Download location
        :return: The path to the downloaded file
        """
        ydl_opts = {
            'outtmpl': f"{output_dir}/%(title)s.%(ext)s",
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
        elif model == "high-definition":
            # Best video+audio
            ydl_opts.update({
                'format': 'bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',
            })
        else:
            raise ValueError("model must be 'mp3' or 'high-definition'")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.download([video_url])
            return ydl.prepare_filename(ydl.extract_info(video_url, download=True)).replace('.webm', '.mp3')
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

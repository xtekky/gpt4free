from __future__ import annotations

import re
from typing import Union
from abc import abstractmethod
from urllib.parse import quote_plus, unquote_plus

def quote_url(url: str) -> str:
    url = unquote_plus(url)
    url = url.split("//", maxsplit=1)
    # If there is no "//" in the URL, then it is a relative URL
    if len(url) == 1:
        return quote_plus(url[0], '/?&=#')
    url[1] = url[1].split("/", maxsplit=1)
    # If there is no "/" after the domain, then it is a domain URL
    if len(url[1]) == 1:
        return url[0] + "//" + url[1][0]
    return url[0] + "//" + url[1][0] + "/" + quote_plus(url[1][1], '/?&=#')

def quote_title(title: str) -> str:
    if title:
        return title.replace("\n", "").replace('"', '')
    return ""

def format_link(url: str, title: str = None) -> str:
    if title is None:
        title = unquote_plus(url.split("//", maxsplit=1)[1].split("?")[0].replace("www.", ""))
    return f"[{quote_title(title)}]({quote_url(url)})"

def format_image(image: str, alt: str, preview: str = None) -> str:
    """
    Formats the given image as a markdown string.

    Args:
        image: The image to format.
        alt (str): The alt for the image.
        preview (str, optional): The preview URL format. Defaults to "{image}?w=200&h=200".

    Returns:
        str: The formatted markdown string.
    """
    return f"[![{quote_title(alt)}]({quote_url(preview.replace('{image}', image) if preview else image)})]({quote_url(image)})"

def format_images_markdown(images: Union[str, list], alt: str, preview: Union[str, list] = None) -> str:
    """
    Formats the given images as a markdown string.

    Args:
        images: The images to format.
        alt (str): The alt for the images.
        preview (str, optional): The preview URL format. Defaults to "{image}?w=200&h=200".

    Returns:
        str: The formatted markdown string.
    """
    if isinstance(images, list) and len(images) == 1:
        images = images[0]
    if isinstance(images, str):
        result = format_image(images, alt, preview)
    else:
        result = "\n".join(
            format_image(image, f"#{idx+1} {alt}", preview[idx] if isinstance(preview, list) else preview)
            for idx, image in enumerate(images)
        )
    start_flag = "<!-- generated images start -->\n"
    end_flag = "<!-- generated images end -->\n"
    return f"\n{start_flag}{result}\n{end_flag}\n"

class ResponseType:
    @abstractmethod
    def __str__(self) -> str:
        pass

class JsonMixin:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        }

class FinishReason(ResponseType, JsonMixin):
    def __init__(self, reason: str, actions: list[str] = None) -> None:
        self.reason = reason
        self.actions = actions

    def __str__(self) -> str:
        return ""

class ToolCalls(ResponseType):
    def __init__(self, list: list):
        self.list = list

    def __str__(self) -> str:
        return ""

    def get_list(self) -> list:
        return self.list

class Usage(ResponseType, JsonMixin):
    def __str__(self) -> str:
        return ""

class TitleGeneration(ResponseType):
    def __init__(self, title: str) -> None:
        self.title = title

    def __str__(self) -> str:
        return ""

class Sources(ResponseType):
    def __init__(self, sources: list[dict[str, str]]) -> None:
        self.list = []
        for source in sources:
            self.add_source(source)

    def add_source(self, source: dict[str, str]):
        url = source.get("url", source.get("link", None))
        if url is not None:
            url = re.sub(r"[&?]utm_source=.+", "", url)
            source["url"] = url
            self.list.append(source)

    def __str__(self) -> str:
        return "\n\n" + ("\n".join([
            f"{idx+1}. {format_link(link['url'], link.get('title', None))}"
            for idx, link in enumerate(self.list)
        ]))

class BaseConversation(ResponseType):
    def __str__(self) -> str:
        return ""

class JsonConversation(BaseConversation, JsonMixin):
    pass

class SynthesizeData(ResponseType, JsonMixin):
    def __init__(self, provider: str, data: dict):
        self.provider = provider
        self.data = data

    def __str__(self) -> str:
        return ""

class RequestLogin(ResponseType):
    def __init__(self, label: str, login_url: str) -> None:
        self.label = label
        self.login_url = login_url

    def __str__(self) -> str:
        return format_link(self.login_url, f"[Login to {self.label}]") + "\n\n"

class ImageResponse(ResponseType):
    def __init__(
        self,
        images: Union[str, list],
        alt: str,
        options: dict = {}
    ):
        self.images = images
        self.alt = alt
        self.options = options

    def __str__(self) -> str:
        return format_images_markdown(self.images, self.alt, self.get("preview"))

    def get(self, key: str):
        return self.options.get(key)

    def get_list(self) -> list[str]:
        return [self.images] if isinstance(self.images, str) else self.images

class ImagePreview(ImageResponse):
    def __str__(self):
        return ""

    def to_string(self):
        return super().__str__()

class Parameters(ResponseType, JsonMixin):
    pass
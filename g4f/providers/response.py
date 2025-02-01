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
        title = title.strip()
        title = " ".join(title.split())
        return title.replace('[', '').replace(']', '')
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
        raise NotImplementedError

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

    def reset(self):
        self.__dict__ = {}

class HiddenResponse(ResponseType):
    def __str__(self) -> str:
        return ""

class FinishReason(JsonMixin, HiddenResponse):
    def __init__(self, reason: str) -> None:
        self.reason = reason

class ToolCalls(HiddenResponse):
    def __init__(self, list: list):
        self.list = list

    def get_list(self) -> list:
        return self.list

class Usage(JsonMixin, HiddenResponse):
    pass

class AuthResult(JsonMixin, HiddenResponse):
    pass

class TitleGeneration(HiddenResponse):
    def __init__(self, title: str) -> None:
        self.title = title

class DebugResponse(JsonMixin, HiddenResponse):
    @classmethod
    def from_dict(cls, data: dict) -> None:
        return cls(**data)

    @classmethod
    def from_str(cls, data: str) -> None:
        return cls(error=data)

class Notification(ResponseType):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"{self.message}\n"

class Reasoning(ResponseType):
    def __init__(
            self,
            token: str = None,
            status: str = None,
            is_thinking: str = None
        ) -> None:
        self.token = token
        self.status = status
        self.is_thinking = is_thinking

    def __str__(self) -> str:
        if self.is_thinking is not None:
            return self.is_thinking
        if self.token is not None:
            return self.token
        if self.status is not None:
            return f"{self.status}\n"
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

class SynthesizeData(HiddenResponse, JsonMixin):
    def __init__(self, provider: str, data: dict):
        self.provider = provider
        self.data = data

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

class PreviewResponse(HiddenResponse):
    def __init__(self, data: str):
        self.data = data

    def to_string(self):
        return self.data

class Parameters(ResponseType, JsonMixin):
    def __str__(self):
        return ""

class ProviderInfo(JsonMixin, HiddenResponse):
    pass
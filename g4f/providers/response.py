from __future__ import annotations

import re
import base64
from typing import Union, Dict, List, Optional
from abc import abstractmethod
from urllib.parse import quote_plus, unquote_plus

def quote_url(url: str) -> str:
    """
    Quote parts of a URL while preserving the domain structure.
    
    Args:
        url: The URL to quote
        
    Returns:
        str: The properly quoted URL
    """
    # Only unquote if needed to avoid double-unquoting
    if '%' in url:
        url = unquote_plus(url)
    
    url_parts = url.split("//", maxsplit=1)
    # If there is no "//" in the URL, then it is a relative URL
    if len(url_parts) == 1:
        return quote_plus(url_parts[0], '/?&=#')
        
    protocol, rest = url_parts
    domain_parts = rest.split("/", maxsplit=1)
    # If there is no "/" after the domain, then it is a domain URL
    if len(domain_parts) == 1:
        return f"{protocol}//{domain_parts[0]}"
    
    domain, path = domain_parts
    return f"{protocol}//{domain}/{quote_plus(path, '/?&=#')}"

def quote_title(title: str) -> str:
    """
    Normalize whitespace in a title.
    
    Args:
        title: The title to normalize
        
    Returns:
        str: The title with normalized whitespace
    """
    return " ".join(title.split()) if title else ""

def format_link(url: str, title: Optional[str] = None) -> str:
    """
    Format a URL and title as a markdown link.
    
    Args:
        url: The URL to link to
        title: The title to display. If None, extracts from URL
        
    Returns:
        str: The formatted markdown link
    """
    if title is None:
        try:
            title = unquote_plus(url.split("//", maxsplit=1)[1].split("?")[0].replace("www.", ""))
        except IndexError:
            title = url
    return f"[{quote_title(title)}]({quote_url(url)})"

def format_image(image: str, alt: str, preview: Optional[str] = None) -> str:
    """
    Formats the given image as a markdown string.

    Args:
        image: The image to format.
        alt: The alt text for the image.
        preview: The preview URL format. Defaults to the original image.

    Returns:
        str: The formatted markdown string.
    """
    preview_url = preview.replace('{image}', image) if preview else image
    return f"[![{quote_title(alt)}]({quote_url(preview_url)})]({quote_url(image)})"

def format_images_markdown(images: Union[str, List[str]], alt: str, 
                           preview: Union[str, List[str]] = None) -> str:
    """
    Formats the given images as a markdown string.

    Args:
        images: The image or list of images to format.
        alt: The alt text for the images.
        preview: The preview URL format or list of preview URLs.
            If not provided, original images are used.

    Returns:
        str: The formatted markdown string.
    """
    if isinstance(images, list) and len(images) == 1:
        images = images[0]
        
    if isinstance(images, str):
        result = format_image(images, alt, preview)
    else:
        result = "\n".join(
            format_image(
                image, 
                f"#{idx+1} {alt}", 
                preview[idx] if isinstance(preview, list) and idx < len(preview) else preview
            )
            for idx, image in enumerate(images)
        )
    
    start_flag = "<!-- generated images start -->\n"
    end_flag = "<!-- generated images end -->\n"
    return f"\n{start_flag}{result}\n{end_flag}\n"

class ResponseType:
    @abstractmethod
    def __str__(self) -> str:
        """Convert the response to a string representation."""
        raise NotImplementedError

class JsonMixin:
    def __init__(self, **kwargs) -> None:
        """Initialize with keyword arguments as attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_dict(self) -> Dict:
        """Return a dictionary of non-private attributes."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        }

    def reset(self) -> None:
        """Reset all attributes."""
        self.__dict__ = {}

class RawResponse(ResponseType, JsonMixin):
    pass

class HiddenResponse(ResponseType):
    def __str__(self) -> str:
        """Hidden responses return an empty string."""
        return ""

class FinishReason(JsonMixin, HiddenResponse):
    def __init__(self, reason: str) -> None:
        """Initialize with a reason."""
        self.reason = reason

class ToolCalls(HiddenResponse):
    def __init__(self, list: List) -> None:
        """Initialize with a list of tool calls."""
        self.list = list

    def get_list(self) -> List:
        """Return the list of tool calls."""
        return self.list

class Usage(JsonMixin, HiddenResponse):
    pass

class AuthResult(JsonMixin, HiddenResponse):
    pass

class TitleGeneration(HiddenResponse):
    def __init__(self, title: str) -> None:
        """Initialize with a title."""
        self.title = title

class DebugResponse(HiddenResponse):
    def __init__(self, log: str) -> None:
        """Initialize with a log message."""
        self.log = log

class Reasoning(ResponseType):
    def __init__(
            self,
            token: Optional[str] = None,
            status: Optional[str] = None,
            is_thinking: Optional[str] = None
        ) -> None:
        """Initialize with token, status, and thinking state."""
        self.token = token
        self.status = status
        self.is_thinking = is_thinking

    def __str__(self) -> str:
        """Return string representation based on available attributes."""
        if self.is_thinking is not None:
            return self.is_thinking
        if self.token is not None:
            return self.token
        if self.status is not None:
            return f"{self.status}\n"
        return ""

    def get_dict(self) -> Dict:
        """Return a dictionary representation of the reasoning."""
        if self.is_thinking is None:
            if self.status is None:
                return {"token": self.token}
            return {"token": self.token, "status": self.status}
        return {"token": self.token, "status": self.status, "is_thinking": self.is_thinking}

class Sources(ResponseType):
    def __init__(self, sources: List[Dict[str, str]]) -> None:
        """Initialize with a list of source dictionaries."""
        self.list = []
        for source in sources:
            self.add_source(source)

    def add_source(self, source: Union[Dict[str, str], str]) -> None:
        """Add a source to the list, cleaning the URL if necessary."""
        source = source if isinstance(source, dict) else {"url": source}
        url = source.get("url", source.get("link", None))
        if url is not None:
            url = re.sub(r"[&?]utm_source=.+", "", url)
            source["url"] = url
            self.list.append(source)

    def __str__(self) -> str:
        """Return formatted sources as a string."""
        if not self.list:
            return ""
        return "\n\n\n\n" + ("\n>\n".join([
            f"> [{idx}] {format_link(link['url'], link.get('title', None))}"
            for idx, link in enumerate(self.list)
        ]))

class YouTube(HiddenResponse):
    def __init__(self, ids: List[str]) -> None:
        """Initialize with a list of YouTube IDs."""
        self.ids = ids

    def to_string(self) -> str:
        """Return YouTube embeds as a string."""
        if not self.ids:
            return ""
        return "\n\n" + ("\n".join([
            f'<iframe type="text/html" src="https://www.youtube.com/embed/{id}"></iframe>'
            for id in self.ids
        ]))

class Audio(ResponseType):
    def __init__(self, data: bytes) -> None:
        """Initialize with audio data bytes."""
        self.data = data

    def __str__(self) -> str:
        """Return audio data as a base64-encoded data URI."""
        data_base64 = base64.b64encode(self.data).decode()
        return f"data:audio/mpeg;base64,{data_base64}"

class BaseConversation(ResponseType):
    def __str__(self) -> str:
        """Return an empty string by default."""
        return ""

class JsonConversation(BaseConversation, JsonMixin):
    pass

class SynthesizeData(HiddenResponse, JsonMixin):
    def __init__(self, provider: str, data: Dict) -> None:
        """Initialize with provider and data."""
        self.provider = provider
        self.data = data

class RequestLogin(HiddenResponse):
    def __init__(self, label: str, login_url: str) -> None:
        """Initialize with label and login URL."""
        self.label = label
        self.login_url = login_url

    def to_string(self) -> str:
        """Return formatted login link as a string."""
        return format_link(self.login_url, f"[Login to {self.label}]") + "\n\n"

class ImageResponse(ResponseType):
    def __init__(
        self,
        images: Union[str, List[str]],
        alt: str,
        options: Dict = {}
    ) -> None:
        """Initialize with images, alt text, and options."""
        self.images = images
        self.alt = alt
        self.options = options

    def __str__(self) -> str:
        """Return images as markdown."""
        return format_images_markdown(self.images, self.alt, self.get("preview"))

    def get(self, key: str) -> any:
        """Get an option value by key."""
        return self.options.get(key)

    def get_list(self) -> List[str]:
        """Return images as a list."""
        return [self.images] if isinstance(self.images, str) else self.images

class ImagePreview(ImageResponse):
    def __str__(self) -> str:
        """Return an empty string for preview."""
        return ""

    def to_string(self) -> str:
        """Return images as markdown."""
        return super().__str__()

class PreviewResponse(HiddenResponse):
    def __init__(self, data: str) -> None:
        """Initialize with data."""
        self.data = data

    def to_string(self) -> str:
        """Return data as a string."""
        return self.data

class Parameters(ResponseType, JsonMixin):
    def __str__(self) -> str:
        """Return an empty string."""
        return ""

class ProviderInfo(JsonMixin, HiddenResponse):
    pass

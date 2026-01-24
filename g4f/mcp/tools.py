"""MCP Tools for gpt4free

This module provides MCP tool implementations that wrap gpt4free capabilities:
- WebSearchTool: Web search using ddg search
- WebScrapeTool: Web page scraping and content extraction
- ImageGenerationTool: Image generation using various AI providers
"""

from __future__ import annotations

from typing import Any, Dict
from abc import ABC, abstractmethod
import urllib.parse

from aiohttp import ClientSession


class MCPTool(ABC):
    """Base class for MCP tools"""
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for tool input parameters"""
        pass
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments
        
        Args:
            arguments: Tool input arguments matching the input_schema
            
        Returns:
            Dict containing either results or an error key with error message
        """
        pass


class WebSearchTool(MCPTool):
    """Web search tool using gpt4free's search capabilities"""
    
    @property
    def description(self) -> str:
        return "Search the web for information using DuckDuckGo. Returns search results with titles, URLs, and snippets."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                },
                "region": {
                    "type": "string",
                    "description": "Search region (default: en-us)"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search
        
        Returns:
            Dict[str, Any]: Search results or error message
        """
        from ..Provider.search.CachedSearch import CachedSearch
        
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        region = arguments.get("region", "en-us")
        
        if not query:
            return {
                "error": "Query parameter is required"
            }
        
        try:
            # Perform search - query parameter is used for search execution
            # and prompt parameter holds the content to be searched
            search_results = await anext(CachedSearch.create_async_generator(
                "",
                [],
                prompt=query,
                max_results=max_results,
                region=region
            ))
            
            return {
                "query": query,
                **search_results.get_dict()
            }
        
        except Exception as e:
            return {
                "error": f"Search failed: {str(e)}"
            }


class WebScrapeTool(MCPTool):
    """Web scraping tool using gpt4free's scraping capabilities"""
    
    @property
    def description(self) -> str:
        return "Scrape and extract text content from a web page URL. Returns cleaned text content with optional word limit."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to scrape"
                },
                "max_words": {
                    "type": "integer",
                    "description": "Maximum number of words to extract (default: 1000)",
                    "default": 1000
                }
            },
            "required": ["url"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web scraping
        
        Returns:
            Dict[str, Any]: Scraped content or error message
        """
        from ..tools.fetch_and_scrape import fetch_and_scrape
        from aiohttp import ClientSession
        
        url = arguments.get("url", "")
        max_words = arguments.get("max_words", 1000)
        
        if not url:
            return {
                "error": "URL parameter is required"
            }
        
        try:
            # Scrape the URL
            async with ClientSession() as session:
                content = await fetch_and_scrape(
                    session=session,
                    url=url,
                    max_words=max_words,
                    add_metadata=True
                )
            
            if not content:
                return {
                    "error": "Failed to scrape content from URL"
                }
            
            return {
                "url": url,
                "content": content,
                "word_count": len(content.split())
            }
        
        except Exception as e:
            return {
                "error": f"Scraping failed: {str(e)}"
            }


class ImageGenerationTool(MCPTool):
    """Image generation tool using gpt4free's image generation capabilities"""
    
    @property
    def description(self) -> str:
        return "Generate images from text prompts using AI image generation providers. Returns a URL to the generated image."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt describing the image to generate"
                },
                "model": {
                    "type": "string",
                    "description": "The image generation model to use (default: flux)",
                    "default": "flux"
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels (default: 1024)",
                    "default": 1024
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels (default: 1024)",
                    "default": 1024
                }
            },
            "required": ["prompt"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image generation
        
        Returns:
            Dict[str, Any]: Generated image data or error message
        """
        from ..client import AsyncClient
        
        prompt = arguments.get("prompt", "")
        model = arguments.get("model", "flux")
        width = arguments.get("width", 1024)
        height = arguments.get("height", 1024)
        
        if not prompt:
            return {
                "error": "Prompt parameter is required"
            }
        
        try:
            # Generate image using gpt4free client
            client = AsyncClient()
            
            response = await client.images.generate(
                model=model,
                prompt=prompt,
                width=width,
                height=height
            )
            
            # Get the image data with proper validation
            if not response:
                return {
                    "error": "Image generation failed: No response from provider"
                }
            
            if not hasattr(response, 'data') or not response.data:
                return {
                    "error": "Image generation failed: No image data in response"
                }
            
            if len(response.data) == 0:
                return {
                    "error": "Image generation failed: Empty image data array"
                }
            
            image_data = response.data[0]
            
            # Check if image_data has url attribute
            if not hasattr(image_data, 'url'):
                return {
                    "error": "Image generation failed: No URL in image data"
                }
            
            image_url = image_data.url

            template = 'Display the image using this template: <a href="{image}" data-width="{width}" data-height="{height}"><img src="{image}" alt="{prompt}"></a>'
            
            # Return result based on URL type
            if image_url.startswith('data:'):
                return {
                    "prompt": prompt,
                    "model": model,
                    "width": width,
                    "height": height,
                    "image": image_url,
                    "template": template
                }
            else:
                if arguments.get("origin") and image_url.startswith("/media/"):
                    image_url = f"{arguments.get('origin')}{image_url}"
                return {
                    "prompt": prompt,
                    "model": model,
                    "width": width,
                    "height": height,
                    "image_url": image_url,
                    "template": template
                }
        
        except Exception as e:
            return {
                "error": f"Image generation failed: {str(e)}"
            }

class MarkItDownTool(MCPTool):
    """MarkItDown tool for converting URLs to markdown format"""
    
    @property
    def description(self) -> str:
        return "Convert a URL to markdown format using MarkItDown. Supports HTTP/HTTPS URLs and returns formatted markdown content."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to convert to markdown format (must be HTTP/HTTPS)"
                },
                "max_content_length": {
                    "type": "integer",
                    "description": "Maximum content length for processing (default: 10000)",
                    "default": 10000
                }
            },
            "required": ["url"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MarkItDown conversion
        
        Returns:
            Dict[str, Any]: Markdown content or error message
        """
        try:
            from ..integration.markitdown import MarkItDown
        except ImportError as e:
            return {
                "error": f"MarkItDown is not installed: {str(e)}"
            }
        
        url = arguments.get("url", "")
        max_content_length = arguments.get("max_content_length", 10000)
        
        if not url:
            return {
                "error": "URL parameter is required"
            }
        
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            return {
                "error": "URL must start with http:// or https://"
            }
        
        try:
            # Initialize MarkItDown
            md = MarkItDown()
            
            # Convert URL to markdown
            result = md.convert_url(url)
            
            if not result:
                return {
                    "error": "Failed to convert URL to markdown"
                }
            
            # Truncate if content exceeds max length
            if len(result) > max_content_length:
                result = result[:max_content_length] + "\n\n[Content truncated...]"
            
            return {
                "url": url,
                "markdown_content": result,
                "content_length": len(result),
                "truncated": len(result) > max_content_length
            }
        
        except Exception as e:
            return {
                "error": f"MarkItDown conversion failed: {str(e)}"
            }

class TextToAudioTool(MCPTool):
    """TextToAudio tool for generating audio from text prompts using Pollinations AI"""
    
    @property
    def description(self) -> str:
        return "Generate an audio URL from a text prompt using Pollinations AI text-to-speech service. Returns a direct URL to the generated audio file."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt to the audio model (example: 'Read this: Hello, world!')"
                },
                "voice": {
                    "type": "string",
                    "description": "Voice option for text-to-speech (default: 'alloy')",
                    "default": "alloy"
                },
                "url_encode": {
                    "type": "boolean",
                    "description": "Whether to URL-encode the prompt text (default: True)",
                    "default": True
                }
            },
            "required": ["prompt"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text-to-speech conversion
        
        Returns:
            Dict[str, Any]: Audio URL or error message
        """
        prompt = arguments.get("prompt", "")
        voice = arguments.get("voice", "alloy")
        url_encode = arguments.get("url_encode", True)
        
        if not prompt:
            return {
                "error": "Prompt parameter is required"
            }
        
        # Validate prompt length (reasonable limit for text-to-speech)
        if len(prompt) > 10000:
            return {
                "error": "Prompt is too long (max 10000 characters)"
            }
        
        try:
            # Prepare the prompt for URL
            if url_encode:
                encoded_prompt = urllib.parse.quote(prompt)
            else:
                encoded_prompt = prompt.replace(" ", "%20")  # Basic space encoding
            
            # Construct the Pollinations AI text-to-speech URL
            audio_url = f"/backend-api/v2/synthesize/Gemini?text={encoded_prompt}"

            if arguments.get("origin"):
                audio_url = f"{arguments.get('origin')}{audio_url}"
                async with ClientSession() as session:
                    async with session.get(audio_url, max_redirects=0) as resp:
                        audio_url = str(resp.url)

            template = 'Play the audio using this template: <audio controls src="{audio_url}">'
            
            return {
                "prompt": prompt,
                "voice": voice,
                "audio_url": audio_url,
                "template": template
            }
        
        except Exception as e:
            return {
                "error": f"Text-to-speech URL generation failed: {str(e)}"
            }

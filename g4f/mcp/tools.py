"""MCP Tools for gpt4free

This module provides MCP tool implementations that wrap gpt4free capabilities:
- WebSearchTool: Web search using ddg search
- WebScrapeTool: Web page scraping and content extraction
- ImageGenerationTool: Image generation using various AI providers
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from abc import ABC, abstractmethod


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
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments"""
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
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search"""
        from ..tools.web_search import do_search
        
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        
        if not query:
            return {
                "error": "Query parameter is required"
            }
        
        try:
            # Perform search
            result, sources = await do_search(
                prompt=query,
                query=query,
                instructions=""
            )
            
            # Format results
            search_results = []
            if sources:
                for i, source in enumerate(sources[:max_results]):
                    search_results.append({
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "snippet": source.get("snippet", "")
                    })
            
            return {
                "query": query,
                "results": search_results,
                "count": len(search_results)
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
        """Execute web scraping"""
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
                    add_source=True
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
        return "Generate images from text prompts using AI image generation providers. Returns base64-encoded image data."
    
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
        """Execute image generation"""
        from ..client import AsyncClient
        from ..image import to_data_uri
        import base64
        
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
            
            # Get the image data
            if response and hasattr(response, 'data') and response.data:
                image_data = response.data[0]
                
                # Convert to base64 if needed
                if hasattr(image_data, 'url'):
                    image_url = image_data.url
                    
                    # Check if it's already a data URI
                    if image_url.startswith('data:'):
                        return {
                            "prompt": prompt,
                            "model": model,
                            "width": width,
                            "height": height,
                            "image": image_url
                        }
                    else:
                        return {
                            "prompt": prompt,
                            "model": model,
                            "width": width,
                            "height": height,
                            "image_url": image_url
                        }
            
            return {
                "error": "Image generation failed: No image data in response"
            }
        
        except Exception as e:
            return {
                "error": f"Image generation failed: {str(e)}"
            }

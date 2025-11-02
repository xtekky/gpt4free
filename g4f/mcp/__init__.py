"""MCP (Model Context Protocol) Server for gpt4free

This module provides an MCP server implementation that exposes gpt4free capabilities
through the Model Context Protocol standard, allowing AI assistants to access:
- Web search functionality
- Web scraping capabilities  
- Image generation using various providers
"""

from .server import MCPServer
from .tools import MarkItDownTool, TextToAudioTool, WebSearchTool, WebScrapeTool, ImageGenerationTool

__all__ = ['MCPServer', 'MarkItDownTool', 'TextToAudioTool', 'WebSearchTool', 'WebScrapeTool', 'ImageGenerationTool']

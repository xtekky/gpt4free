"""MCP (Model Context Protocol) Server for gpt4free

This module provides an MCP server implementation that exposes gpt4free capabilities
through the Model Context Protocol standard, allowing AI assistants to access:
- Web search functionality
- Web scraping capabilities
- Image generation using various providers
- Safe Python code execution
- Workspace file management (read, write, list, delete) at ~/.g4f/workspace
- .pa.py custom provider loading and execution
"""

from .server import MCPServer
from .tools import (
    MarkItDownTool,
    TextToAudioTool,
    WebSearchTool,
    ImageGenerationTool,
    PythonExecuteTool,
    FileReadTool,
    FileListTool,
    FileDeleteTool,
    CreateDirectoryTool,
    CreateFileTool,
    FetchWebpageTool,
    FileSearchGlobTool,
    GrepSearchTool,
    GithubRepoTool,
    GithubTextSearchTool,
)
from .pa_provider import (
    execute_safe_code,
    load_pa_provider,
    list_pa_providers,
    get_workspace_dir,
    get_pa_registry,
    SAFE_MODULES,
    SafeExecutionResult,
    PaProviderRegistry,
)

__all__ = [
    'MCPServer',
    # Original tools
    'MarkItDownTool',
    'TextToAudioTool',
    'WebSearchTool',
    'ImageGenerationTool',
    # New tools
    'PythonExecuteTool',
    'FileReadTool',
    'FileListTool',
    'FileDeleteTool',
    'CreateDirectoryTool',
    'CreateFileTool',
    'FetchWebpageTool',
    'FileSearchGlobTool',
    'GrepSearchTool',
    'GithubRepoTool',
    'GithubTextSearchTool',
    # PA provider system
    'execute_safe_code',
    'load_pa_provider',
    'list_pa_providers',
    'get_workspace_dir',
    'get_pa_registry',
    'SAFE_MODULES',
    'SafeExecutionResult',
    'PaProviderRegistry',
]

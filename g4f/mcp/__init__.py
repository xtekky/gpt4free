"""MCP (Model Context Protocol) Server for gpt4free

This module provides an MCP server implementation that exposes gpt4free capabilities
through the Model Context Protocol standard, allowing AI assistants to access:
- Web search functionality
- Web scraping capabilities
- Image generation using various providers
- Safe Python code execution
- Shell / terminal command execution (host mode only — disabled automatically
  when safe mode is active, e.g. inside detected sandboxes/containers)
- Workspace file management (read, write, list, delete) in the workspace
  directory (``~/.g4f/workspace`` by default, overridable with ``G4F_DIR``)
- .pa.py custom provider loading and execution
"""

from .server import MCPServer
from .sandbox import (
    detect_sandbox,
    get_sandbox_reasons,
    safe_mode_enabled,
    shell_tool_enabled,
)
from .tools import (
    MarkItDownTool,
    TextToAudioTool,
    WebSearchTool,
    ImageGenerationTool,
    PythonExecuteTool,
    ShellExecuteTool,
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
    TokenOptimizerTool,
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
    "MCPServer",
    # Original tools
    "MarkItDownTool",
    "TextToAudioTool",
    "WebSearchTool",
    "ImageGenerationTool",
    # New tools
    "PythonExecuteTool",
    "ShellExecuteTool",
    "FileReadTool",
    "FileListTool",
    "FileDeleteTool",
    "CreateDirectoryTool",
    "CreateFileTool",
    "FetchWebpageTool",
    "FileSearchGlobTool",
    "GrepSearchTool",
    "GithubRepoTool",
    "GithubTextSearchTool",
    "TokenOptimizerTool",
    # Sandbox detection / safe mode
    "detect_sandbox",
    "get_sandbox_reasons",
    "safe_mode_enabled",
    "shell_tool_enabled",
    # PA provider system
    "execute_safe_code",
    "load_pa_provider",
    "list_pa_providers",
    "get_workspace_dir",
    "get_pa_registry",
    "SAFE_MODULES",
    "SafeExecutionResult",
    "PaProviderRegistry",
]

"""MCP Tools for gpt4free

This module provides MCP tool implementations that wrap gpt4free capabilities:
- WebSearchTool: Web search using ddg search
- WebScrapeTool: Web page scraping and content extraction
- ImageGenerationTool: Image generation using various AI providers
- PythonExecuteTool: Safe Python code execution with whitelisted modules
- FileReadTool: Read files from the ~/.g4f/workspace directory (supports startLine/endLine)
- FileSearchTool: Search files and file contents in the workspace
- FileWriteTool: Write files to the ~/.g4f/workspace directory
- FileListTool: List files in the ~/.g4f/workspace directory
- FileDeleteTool: Delete files from the ~/.g4f/workspace directory
"""

from __future__ import annotations

from typing import Any, Dict
from abc import ABC, abstractmethod
import fnmatch
import re
import urllib.parse

from aiohttp import ClientSession


class MCPTool(ABC):
    """Base class for MCP tools"""

    def __init__(self, safe_mode: bool = False) -> None:
        """Initialize tool with optional safe mode.

        Args:
            safe_mode: When ``True`` the tool operates in a restricted mode
                where callers cannot expand the module allowlist and certain
                sensitive listing operations are blocked.
        """
        self.safe_mode = safe_mode

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


class PythonExecuteTool(MCPTool):
    """Safe Python code execution tool with whitelisted module imports.

    Executes the supplied code snippet inside a restricted sandbox where only
    a curated list of modules may be imported and file-system access is limited
    to the ``~/.g4f/workspace`` directory.  The value assigned to the ``result``
    variable (if any) is returned along with captured stdout/stderr.
    """

    @property
    def description(self) -> str:
        return (
            "Execute a Python code snippet safely. Only whitelisted modules may "
            "be imported (math, json, re, datetime, asyncio, aiohttp, g4f, …). "
            "File access is restricted to the ~/.g4f/workspace directory. "
            "Assign the value you want back to a variable named 'result'. "
            "Returns stdout, stderr, and the value of 'result'."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in the safe sandbox",
                },
                "allowed_extra_modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of additional module names to allow "
                        "beyond the default whitelist (ignored in safe mode)"
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Wall-clock seconds to allow before aborting execution "
                        f"(max {30.0}s; ignored in safe mode)"
                    ),
                },
                "max_depth": {
                    "type": "integer",
                    "description": (
                        "Maximum Python call-stack depth inside the sandbox "
                        f"(max {500}; ignored in safe mode)"
                    ),
                },
            },
            "required": ["code"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import execute_safe_code, SAFE_MODULES, MAX_EXEC_TIMEOUT, MAX_RECURSION_DEPTH

        code = arguments.get("code", "")
        if not code:
            return {"error": "code parameter is required"}

        if self.safe_mode:
            # In safe mode the caller cannot override any security parameters
            allowed = SAFE_MODULES
            timeout = MAX_EXEC_TIMEOUT
            max_depth = MAX_RECURSION_DEPTH
        else:
            extra_names = arguments.get("allowed_extra_modules") or []
            allowed = SAFE_MODULES | frozenset(extra_names)
            # Allow callers to reduce (but not exceed) the defaults
            requested_timeout = arguments.get("timeout")
            if requested_timeout is not None:
                try:
                    timeout = min(float(requested_timeout), MAX_EXEC_TIMEOUT)
                except (TypeError, ValueError):
                    return {"error": "timeout must be a number"}
            else:
                timeout = MAX_EXEC_TIMEOUT
            requested_depth = arguments.get("max_depth")
            if requested_depth is not None:
                try:
                    max_depth = min(int(requested_depth), MAX_RECURSION_DEPTH)
                except (TypeError, ValueError):
                    return {"error": "max_depth must be an integer"}
            else:
                max_depth = MAX_RECURSION_DEPTH

        try:
            exec_result = execute_safe_code(
                code,
                allowed_modules=allowed,
                timeout=timeout,
                max_depth=max_depth,
            )
            return exec_result.to_dict()
        except Exception as exc:
            return {"error": f"Execution error: {exc}"}


class FileReadTool(MCPTool):
    """Read a file from the ``~/.g4f/workspace`` directory."""

    @property
    def description(self) -> str:
        return (
            "Read the text content of a file inside the ~/.g4f/workspace directory. "
            "Provide a relative path from the workspace root. "
            "Optionally specify startLine and endLine (1-based) to read only a range of lines."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file inside the workspace",
                },
                "startLine": {
                    "type": "number",
                    "description": "The line number to start reading from, 1-based. If omitted, reads from the beginning.",
                },
                "endLine": {
                    "type": "number",
                    "description": "The inclusive line number to end reading at, 1-based. If omitted, reads to the end of file.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir

        rel_path = arguments.get("path", "")
        if not rel_path:
            return {"error": "path parameter is required"}

        start_line = arguments.get("startLine")
        end_line = arguments.get("endLine")

        if start_line is not None:
            try:
                start_line = int(start_line)
            except (TypeError, ValueError):
                return {"error": "startLine must be a number"}
            if start_line < 1:
                return {"error": "startLine must be >= 1"}

        if end_line is not None:
            try:
                end_line = int(end_line)
            except (TypeError, ValueError):
                return {"error": "endLine must be a number"}
            if start_line is not None and end_line < start_line:
                return {"error": "endLine must be >= startLine"}

        workspace = get_workspace_dir().resolve()
        try:
            target = (workspace / rel_path).resolve()
            if not str(target).startswith(str(workspace)):
                return {"error": "Access outside the workspace is not allowed"}
            if not target.exists():
                return {"error": f"File not found: {rel_path}"}
            if not target.is_file():
                return {"error": f"Path is not a file: {rel_path}"}

            if start_line is not None or end_line is not None:
                lines = target.read_text(encoding="utf-8").splitlines(True)
                total_lines = len(lines)
                s = (start_line - 1) if start_line is not None else 0
                e = end_line if end_line is not None else total_lines
                s = max(0, min(s, total_lines))
                e = max(s, min(e, total_lines))
                content = "".join(lines[s:e])
                return {
                    "path": rel_path,
                    "content": content,
                    "size": len(content),
                    "startLine": s + 1,
                    "endLine": e,
                    "totalLines": total_lines,
                }

            content = target.read_text(encoding="utf-8")
            return {
                "path": rel_path,
                "content": content,
                "size": len(content),
            }
        except Exception as exc:
            return {"error": f"Read failed: {exc}"}


class FileListTool(MCPTool):
    """List files and directories inside the ``~/.g4f/workspace`` directory."""

    @property
    def description(self) -> str:
        return (
            "List files and directories inside the ~/.g4f/workspace directory. "
            "Optionally provide a subdirectory path relative to the workspace root. "
            "Returns names, types, and sizes."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path to a subdirectory inside the workspace "
                        "(default: workspace root)"
                    ),
                    "default": "",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list files recursively (default: false)",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir, is_hidden_file

        rel_path = arguments.get("path", "") or ""
        recursive = bool(arguments.get("recursive", False))

        workspace = get_workspace_dir().resolve()
        try:
            target = (workspace / rel_path).resolve() if rel_path else workspace
            if not str(target).startswith(str(workspace)):
                return {"error": "Access outside the workspace is not allowed"}
            if self.safe_mode and target == workspace:
                return {"error": "Listing the workspace root directory is not allowed in safe mode"}
            if not target.exists():
                return {"error": f"Directory not found: {rel_path or '/'}"}
            if not target.is_dir():
                return {"error": f"Path is not a directory: {rel_path}"}

            entries = []
            skipped = 0
            iterator = target.rglob("*") if recursive else target.iterdir()
            for entry in sorted(iterator):
                try:
                    if is_hidden_file(entry):
                        continue
                    rel = str(entry.relative_to(workspace))
                    info: Dict[str, Any] = {
                        "path": rel,
                        "type": "file" if entry.is_file() else "directory",
                    }
                    if entry.is_file():
                        info["size"] = entry.stat().st_size
                    entries.append(info)
                except Exception:
                    skipped += 1
                    continue

            result: Dict[str, Any] = {
                "workspace": "" if self.safe_mode else str(workspace),
                "path": rel_path or "/",
                "entries": entries,
                "count": len(entries),
            }
            if skipped:
                result["skipped"] = skipped
            return result
        except Exception as exc:
            return {"error": f"List failed: {exc}"}


class FileDeleteTool(MCPTool):
    """Delete a file from the ``~/.g4f/workspace`` directory."""

    @property
    def description(self) -> str:
        return (
            "Delete a file from the ~/.g4f/workspace directory. "
            "Provide a relative path from the workspace root. "
            "Only files can be deleted; directories are not removed."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file inside the workspace",
                }
            },
            "required": ["path"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir

        rel_path = arguments.get("path", "")
        if not rel_path:
            return {"error": "path parameter is required"}

        workspace = get_workspace_dir().resolve()
        try:
            target = (workspace / rel_path).resolve()
            if not str(target).startswith(str(workspace)):
                return {"error": "Access outside the workspace is not allowed"}
            if not target.exists():
                return {"error": f"File not found: {rel_path}"}
            if not target.is_file():
                return {"error": f"Path is not a file (directories cannot be deleted): {rel_path}"}
            target.unlink()
            return {"path": rel_path, "deleted": True}
        except Exception as exc:
            return {"error": f"Delete failed: {exc}"}

class CreateDirectoryTool(MCPTool):
    """Create a directory (and all parents) inside the ``~/.g4f/workspace`` directory."""

    @property
    def description(self) -> str:
        return (
            "Create a new directory structure in the ~/.g4f/workspace. "
            "Will recursively create all directories in the path, like mkdir -p. "
            "You do not need to use this tool before using create_file, that tool will automatically create the needed directories."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dirPath": {
                    "type": "string",
                    "description": "Relative path to the directory to create inside the workspace",
                }
            },
            "required": ["dirPath"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir

        rel_path = arguments.get("dirPath", "")
        if not rel_path:
            return {"error": "dirPath parameter is required"}

        workspace = get_workspace_dir().resolve()
        try:
            target = (workspace / rel_path).resolve()
            if not str(target).startswith(str(workspace)):
                return {"error": "Access outside the workspace is not allowed"}
            target.mkdir(parents=True, exist_ok=True)
            return {"dirPath": rel_path, "created": True}
        except Exception as exc:
            return {"error": f"Create directory failed: {exc}"}


class CreateFileTool(MCPTool):
    """Create a new file inside the ``~/.g4f/workspace`` directory."""

    @property
    def description(self) -> str:
        return (
            "Create a new file in the ~/.g4f/workspace with the specified content. "
            "The directory will be created if it does not already exist. "
            "Never use this tool to edit a file that already exists."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Relative path to the file to create inside the workspace",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["filePath", "content"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir

        rel_path = arguments.get("filePath", "")
        content = arguments.get("content")

        if not rel_path:
            return {"error": "filePath parameter is required"}
        if content is None:
            return {"error": "content parameter is required"}

        workspace = get_workspace_dir().resolve()
        try:
            target = (workspace / rel_path).resolve()
            if not str(target).startswith(str(workspace)):
                return {"error": "Access outside the workspace is not allowed"}
            if target.exists():
                return {"error": f"File already exists: {rel_path}. Use FileWriteTool to overwrite."}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"filePath": rel_path, "created": True, "size": len(content)}
        except Exception as exc:
            return {"error": f"Create file failed: {exc}"}


class FetchWebpageTool(MCPTool):
    """Fetch and return the main content from one or more web pages."""

    @property
    def description(self) -> str:
        return (
            "Fetches the main content from web pages. Useful for summarizing or analyzing webpage content. "
            "Provide one or more URLs and a query describing what content to find."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "An array of URLs to fetch content from.",
                },
                "query": {
                    "type": "string",
                    "description": "The query to search for in the web page's content.",
                },
                "max_words": {
                    "type": "integer",
                    "description": "Maximum number of words to extract per page (default: 1000)",
                    "default": 1000,
                },
            },
            "required": ["urls", "query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from ..tools.fetch_and_scrape import fetch_and_scrape

        urls = arguments.get("urls", [])
        query = arguments.get("query", "")
        max_words = int(arguments.get("max_words", 1000))

        if not urls:
            return {"error": "urls parameter is required"}
        if not query:
            return {"error": "query parameter is required"}

        results = []
        async with ClientSession() as session:
            for url in urls:
                try:
                    content = await fetch_and_scrape(
                        session=session,
                        url=url,
                        max_words=max_words,
                        add_metadata=True,
                    )
                    results.append({
                        "url": url,
                        "content": content or "",
                        "word_count": len((content or "").split()),
                    })
                except Exception as exc:
                    results.append({"url": url, "error": str(exc)})

        return {"query": query, "results": results}


class FileSearchGlobTool(MCPTool):
    """Search for files in the workspace by glob pattern."""

    @property
    def description(self) -> str:
        return (
            "Search for files in the ~/.g4f/workspace by glob pattern. "
            "Returns the paths of matching files. "
            "Examples: **/*.py to match all Python files; src/** to match all files under src/."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Glob pattern to match file names or paths (e.g. '**/*.py', 'src/**')",
                },
                "maxResults": {
                    "type": "number",
                    "description": "The maximum number of results to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir, is_hidden_file

        pattern = arguments.get("query", "")
        max_results = int(arguments.get("maxResults", 50))

        if not pattern:
            return {"error": "query parameter is required"}

        workspace = get_workspace_dir().resolve()
        try:
            matches = []
            for entry in workspace.rglob("*"):
                if not entry.is_file():
                    continue
                rel = entry.relative_to(workspace)
                rel_str = rel.as_posix()
                if is_hidden_file(rel_str):
                    continue
                if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(entry.name, pattern):
                    matches.append(rel_str)
                if len(matches) >= max_results:
                    break
            return {"pattern": pattern, "matches": matches, "count": len(matches)}
        except Exception as exc:
            return {"error": f"File search failed: {exc}"}


class GrepSearchTool(MCPTool):
    """Fast text search in workspace files."""

    @property
    def description(self) -> str:
        return (
            "Do a fast text search in the ~/.g4f/workspace files. "
            "Supports exact string or regex patterns. "
            "Use includePattern to restrict to specific file types. "
            "Returns matching file paths and line snippets."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The pattern to search for. Use regex alternation (e.g. 'word1|word2') for multiple words.",
                },
                "isRegexp": {
                    "type": "boolean",
                    "description": "Whether the pattern is a regular expression.",
                    "default": False,
                },
                "includePattern": {
                    "type": "string",
                    "description": "Glob pattern to filter which files to search (e.g. '**/*.py').",
                },
                "maxResults": {
                    "type": "number",
                    "description": "Maximum number of matching lines to return (default: 100)",
                    "default": 100,
                },
            },
            "required": ["query", "isRegexp"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir, is_hidden_file

        pattern = arguments.get("query", "")
        is_regexp = bool(arguments.get("isRegexp", False))
        include_pattern = arguments.get("includePattern")
        max_results = int(arguments.get("maxResults", 100))

        if not pattern:
            return {"error": "query parameter is required"}

        workspace = get_workspace_dir().resolve()
        try:
            if is_regexp:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    return {"error": f"Invalid regular expression: {exc}"}
            else:
                compiled = None

            matches = []
            for entry in sorted(workspace.rglob("*")):
                if not entry.is_file():
                    continue
                rel_str = entry.relative_to(workspace).as_posix()
                if is_hidden_file(rel_str):
                    continue
                if include_pattern and not fnmatch.fnmatch(rel_str, include_pattern):
                    continue
                try:
                    lines = entry.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception:
                    continue
                for lineno, line in enumerate(lines, 1):
                    if is_regexp:
                        hit = bool(compiled.search(line))
                    else:
                        hit = pattern.lower() in line.lower()
                    if hit:
                        matches.append({
                            "path": rel_str,
                            "line": lineno,
                            "text": line,
                        })
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break

            return {"pattern": pattern, "matches": matches, "count": len(matches)}
        except Exception as exc:
            return {"error": f"Grep search failed: {exc}"}


class GithubRepoTool(MCPTool):
    """Search a GitHub repository for relevant source code snippets."""

    @property
    def description(self) -> str:
        return (
            "Searches a GitHub repository for relevant source code snippets. "
            "Only use this tool if the user is clearly asking for code from a specific GitHub repository. "
            "Do not use for repos the user has open locally."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "The GitHub repository to search, formatted as '<owner>/<repo>'.",
                },
                "query": {
                    "type": "string",
                    "description": "The query to search for in the repository.",
                },
            },
            "required": ["repo", "query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo = arguments.get("repo", "")
        query = arguments.get("query", "")

        if not repo:
            return {"error": "repo parameter is required"}
        if not query:
            return {"error": "query parameter is required"}
        if "/" not in repo:
            return {"error": "repo must be formatted as '<owner>/<repo>'"}

        try:
            search_url = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}+repo:{urllib.parse.quote(repo)}"
            headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
            async with ClientSession() as session:
                async with session.get(search_url, headers=headers) as resp:
                    if resp.status == 403:
                        return {"error": "GitHub API rate limit exceeded. Try again later."}
                    if resp.status != 200:
                        return {"error": f"GitHub API error: HTTP {resp.status}"}
                    data = await resp.json()

            items = data.get("items", [])[:10]
            results = []
            for item in items:
                results.append({
                    "path": item.get("path"),
                    "url": item.get("html_url"),
                    "repository": item.get("repository", {}).get("full_name"),
                })
            return {"repo": repo, "query": query, "results": results, "count": len(results)}
        except Exception as exc:
            return {"error": f"GitHub repo search failed: {exc}"}


class GithubTextSearchTool(MCPTool):
    """Lexically search a GitHub repository or organisation for files with specific keywords."""

    @property
    def description(self) -> str:
        return (
            "Lexically searches a GitHub repository or organization for files containing "
            "specific keywords or code patterns. Use 'owner/repo' to search a single repository "
            "or an org name (no slash) to search across an entire organization. "
            "Unlike semantic search, this uses keyword matching."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "GitHub scope: 'owner/repo' for a single repo, or an org name for org-wide search.",
                },
                "query": {
                    "type": "string",
                    "description": "Keyword search query. Supports GitHub code search syntax (e.g. 'language:python', 'path:src/').",
                },
                "maxResults": {
                    "type": "number",
                    "description": "Maximum number of search results to return (default: 100)",
                    "default": 100,
                },
            },
            "required": ["scope", "query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        scope = arguments.get("scope", "")
        query = arguments.get("query", "")
        max_results = int(arguments.get("maxResults", 100))

        if not scope:
            return {"error": "scope parameter is required"}
        if not query:
            return {"error": "query parameter is required"}

        try:
            # Determine if scope is a repo or an org
            if "/" in scope:
                qualifier = f"repo:{scope}"
            else:
                qualifier = f"org:{scope}"

            search_url = (
                f"https://api.github.com/search/code"
                f"?q={urllib.parse.quote(query)}+{urllib.parse.quote(qualifier)}"
                f"&per_page={min(max_results, 100)}"
            )
            headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
            async with ClientSession() as session:
                async with session.get(search_url, headers=headers) as resp:
                    if resp.status == 403:
                        return {"error": "GitHub API rate limit exceeded. Try again later."}
                    if resp.status != 200:
                        return {"error": f"GitHub API error: HTTP {resp.status}"}
                    data = await resp.json()

            items = data.get("items", [])[:max_results]
            results = []
            for item in items:
                results.append({
                    "path": item.get("path"),
                    "url": item.get("html_url"),
                    "repository": item.get("repository", {}).get("full_name"),
                    "name": item.get("name"),
                })
            return {"scope": scope, "query": query, "results": results, "count": len(results)}
        except Exception as exc:
            return {"error": f"GitHub text search failed: {exc}"}


class ApplyPatchTool(MCPTool):
    """Apply a unified diff patch to a file or directory using the system 'patch' command."""

    @property
    def description(self) -> str:
        return (
            "Apply a unified diff patch to a target file or directory using the system 'patch' command. "
            "Provide the target path, patch content, and optional parameters for strip level, backup, and dry run. "
            "Returns success status, output from the patch command, and any error messages."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_path": {
                    "type": "string",
                    "description": "Path to the target file or directory to patch"
                },
                "patch_content": {
                    "type": "string",
                    "description": "The unified diff patch content to apply"
                },
                "strip": {
                    "type": "integer",
                    "description": "Number of leading path components to strip from file paths in the patch (default: 1)",
                    "default": 1
                },
                "backup": {
                    "type": "boolean",
                    "description": "Whether to create backup files before patching (default: false)",
                    "default": False
                },
                "dry_run": {
                    "type": "boolean",
                    "description": (
                        "If true, perform a dry run without making changes (default: false). "
                        "The output will indicate whether the patch would apply cleanly."
                    ),
                    "default": False
                },
            },
            "required": ["target_path", "patch_content"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from .pa_provider import get_workspace_dir
        from .apply_patch import apply_patch_with_fallback

        target_path = arguments.get("target_path", "")
        patch_content = arguments.get("patch_content", "")
        backup = bool(arguments.get("backup", False))
        dry_run = bool(arguments.get("dry_run", False))
        workspace = get_workspace_dir().resolve()
        try:
            target = (workspace / target_path).resolve()
            if not str(target).startswith(str(workspace)):
                return {"error": "Access outside the workspace is not allowed"}
            if not target.exists():
                return {"error": f"File not found: {target_path}"}
        except Exception as exc:
            return {"error": f"Invalid target path: {exc}"}

        return apply_patch_with_fallback(patch_content, str(target), backup, dry_run)
#!/usr/bin/env python3
"""
Filesystem Tools for GPT4Free

Provides file system operations that can be called by models through tool calling.
Includes: read, write, list, delete, search, and file operations.

Usage:
    from g4f.tools.filesystem import FileSystemTools

    tools = FileSystemTools.get_tools()
    # Use with tool calling
"""

from __future__ import annotations

import os
import json
import shutil
import fnmatch
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator
from datetime import datetime

from .. import debug
from ..providers.response import ToolCalls


class FileSystemTools:
    """
    File system tools that can be called by models.

    Provides safe file operations with proper error handling.
    """

    # Default safe directories (can be configured)
    SAFE_DIRECTORIES = [
        Path.home() / "g4f_files",
        Path.cwd(),  # Current working directory
        Path.cwd() / "g4f_files",
        Path("/tmp/g4f_files"),
    ]

    # Maximum file size to read (10MB default)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Forbidden patterns for security
    FORBIDDEN_PATTERNS = [
        "*.pyc",
        "__pycache__/*",
        ".git/*",
        "*.exe",
        "*.dll",
        "*.so",
    ]

    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """
        Get the list of available filesystem tools in OpenAI format.

        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file. Use this to view file contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding (default: utf-8)",
                                "default": "utf-8"
                            },
                            "lines": {
                                "type": "object",
                                "description": "Read specific line range",
                                "properties": {
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"}
                                }
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file. Creates the file if it doesn't exist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            },
                            "mode": {
                                "type": "string",
                                "description": "Write mode: 'w' (overwrite) or 'a' (append)",
                                "enum": ["w", "a"],
                                "default": "w"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding (default: utf-8)",
                                "default": "utf-8"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to list"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Filter by pattern (e.g., '*.py', '*.txt')",
                                "default": "*"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "List recursively (default: false)",
                                "default": False
                            },
                            "include_files": {
                                "type": "boolean",
                                "description": "Include files in listing",
                                "default": True
                            },
                            "include_dirs": {
                                "type": "boolean",
                                "description": "Include directories in listing",
                                "default": True
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory or directory tree.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to create"
                            },
                            "parents": {
                                "type": "boolean",
                                "description": "Create parent directories if needed",
                                "default": True
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file. Use with caution!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to delete"
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force deletion without confirmation",
                                "default": False
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_directory",
                    "description": "Delete a directory and its contents. Use with extreme caution!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to delete"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Delete recursively (required for non-empty dirs)",
                                "default": True
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force deletion without confirmation",
                                "default": False
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Move or rename a file or directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Source path"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination path"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Overwrite if destination exists",
                                "default": False
                            }
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a file or directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Source path"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination path"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Copy directories recursively",
                                "default": False
                            }
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files by name or pattern in a directory tree.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Root directory to search in"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Pattern to match (e.g., '*.py', 'test_*')"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Search recursively",
                                "default": True
                            }
                        },
                        "required": ["path", "pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_info",
                    "description": "Get metadata about a file or directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file or directory"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_exists",
                    "description": "Check if a file or directory exists.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to check"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_in_files",
                    "description": "Search for text content within files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory to search in"
                            },
                            "query": {
                                "type": "string",
                                "description": "Text to search for"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "File pattern to filter (e.g., '*.py')",
                                "default": "*"
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Case-sensitive search",
                                "default": False
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Search recursively",
                                "default": True
                            }
                        },
                        "required": ["path", "query"]
                    }
                }
            }
        ]

    @classmethod
    def validate_path(cls, path: str, must_exist: bool = False, allow_write: bool = False) -> Tuple[bool, str]:
        """
        Validate a path for security.

        Args:
            path: Path to validate
            must_exist: Whether the path must already exist
            allow_write: Whether to allow write operations (stricter validation)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path_obj = Path(path).resolve()

            # Check if path is in safe directories using proper path containment
            is_safe = False
            for safe_dir in cls.SAFE_DIRECTORIES:
                try:
                    safe_path = safe_dir.resolve()
                    # Use relative_to for proper path containment check
                    try:
                        path_obj.relative_to(safe_path)
                        is_safe = True
                        break
                    except ValueError:
                        # Also check if it's the safe_dir itself
                        if path_obj == safe_path:
                            is_safe = True
                            break
                except:
                    pass

            # Allow relative paths from current directory using proper containment
            if not is_safe:
                try:
                    cwd = Path.cwd().resolve()
                    try:
                        path_obj.relative_to(cwd)
                        is_safe = True
                    except ValueError:
                        if path_obj == cwd:
                            is_safe = True
                except:
                    pass

            if not is_safe:
                if must_exist or allow_write:
                    return False, f"Path '{path}' is outside safe directories"

            # Check forbidden patterns
            path_str = str(path_obj)
            for pattern in cls.FORBIDDEN_PATTERNS:
                if fnmatch.fnmatch(path_str, pattern):
                    return False, f"Path matches forbidden pattern: {pattern}"

            # Check existence if required
            if must_exist and not path_obj.exists():
                return False, f"Path does not exist: {path}"

            return True, ""

        except Exception as e:
            return False, f"Invalid path: {str(e)}"

    @classmethod
    def read_file(
        cls,
        path: str,
        encoding: str = "utf-8",
        lines: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Read file contents.

        Args:
            path: File path
            encoding: File encoding
            lines: Line range {"start": int, "end": int}

        Returns:
            Result dictionary with success, content, and error
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            # Check file size
            file_size = path_obj.stat().st_size
            if file_size > cls.MAX_FILE_SIZE:
                return {
                    "success": False,
                    "error": f"File too large ({file_size} bytes). Max: {cls.MAX_FILE_SIZE}"
                }

            # Read file
            content = path_obj.read_text(encoding=encoding)

            # Apply line filter if specified
            if lines:
                all_lines = content.splitlines()
                start = lines.get("start", 0)
                end = lines.get("end", len(all_lines))
                content = "\n".join(all_lines[start:end])

            return {
                "success": True,
                "content": content,
                "path": str(path_obj),
                "size": file_size,
                "lines": len(content.splitlines())
            }

        except Exception as e:
            return {"success": False, "error": f"Read error: {str(e)}"}

    @classmethod
    def write_file(
        cls,
        path: str,
        content: str,
        mode: str = "w",
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Write content to file.

        Args:
            path: File path
            content: Content to write
            mode: Write mode ('w' or 'a')
            encoding: File encoding

        Returns:
            Result dictionary
        """
        try:
            # Validate path - allow write operations
            is_valid, error = cls.validate_path(path, must_exist=False, allow_write=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            # Check if file exists BEFORE writing
            file_existed = path_obj.exists()

            # Create parent directories if needed
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with path_obj.open(mode=mode, encoding=encoding) as f:
                f.write(content)

            return {
                "success": True,
                "path": str(path_obj),
                "bytes_written": len(content.encode(encoding)),
                "mode": "created" if not file_existed else "updated"
            }

        except Exception as e:
            return {"success": False, "error": f"Write error: {str(e)}"}

    @classmethod
    def list_directory(
        cls,
        path: str,
        pattern: str = "*",
        recursive: bool = False,
        include_files: bool = True,
        include_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Filter pattern
            recursive: List recursively
            include_files: Include files
            include_dirs: Include directories

        Returns:
            Result dictionary with files and directories
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            if not path_obj.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            files = []
            directories = []

            if recursive:
                # Recursive listing
                for item in path_obj.rglob(pattern):
                    rel_path = item.relative_to(path_obj)
                    if item.is_file() and include_files:
                        files.append(str(rel_path))
                    elif item.is_dir() and include_dirs:
                        directories.append(str(rel_path))
            else:
                # Non-recursive listing
                for item in path_obj.glob(pattern):
                    rel_path = item.relative_to(path_obj)
                    if item.is_file() and include_files:
                        files.append(str(rel_path))
                    elif item.is_dir() and include_dirs:
                        directories.append(str(rel_path))

            return {
                "success": True,
                "path": str(path_obj),
                "files": sorted(files),
                "directories": sorted(directories),
                "total_files": len(files),
                "total_dirs": len(directories)
            }

        except Exception as e:
            return {"success": False, "error": f"List error: {str(e)}"}

    @classmethod
    def create_directory(cls, path: str, parents: bool = True) -> Dict[str, Any]:
        """
        Create a directory.

        Args:
            path: Directory path
            parents: Create parent directories

        Returns:
            Result dictionary
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=False)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            if parents:
                path_obj.mkdir(parents=True, exist_ok=True)
            else:
                path_obj.mkdir()

            return {
                "success": True,
                "path": str(path_obj),
                "created": True
            }

        except Exception as e:
            return {"success": False, "error": f"Create error: {str(e)}"}

    @classmethod
    def delete_file(cls, path: str, force: bool = False) -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            path: File path
            force: Force deletion

        Returns:
            Result dictionary
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            if not path_obj.is_file():
                return {"success": False, "error": f"Not a file: {path}"}

            path_obj.unlink()

            return {
                "success": True,
                "path": str(path_obj),
                "deleted": True
            }

        except Exception as e:
            return {"success": False, "error": f"Delete error: {str(e)}"}

    @classmethod
    def delete_directory(cls, path: str, recursive: bool = True, force: bool = False) -> Dict[str, Any]:
        """
        Delete a directory.

        Args:
            path: Directory path
            recursive: Delete recursively (required for non-empty dirs)
            force: Force deletion

        Returns:
            Result dictionary
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            if not path_obj.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            # Honor recursive parameter
            if recursive:
                shutil.rmtree(path_obj)
            else:
                # Only delete if empty
                if any(path_obj.iterdir()):
                    return {"success": False, "error": "Directory not empty. Use recursive=True"}
                path_obj.rmdir()

            return {
                "success": True,
                "path": str(path_obj),
                "deleted": True
            }

        except Exception as e:
            return {"success": False, "error": f"Delete error: {str(e)}"}

    @classmethod
    def move_file(cls, source: str, destination: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Move or rename a file/directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite if exists

        Returns:
            Result dictionary
        """
        try:
            # Validate paths
            is_valid, error = cls.validate_path(source, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            is_valid, error = cls.validate_path(destination, must_exist=False)
            if not is_valid:
                return {"success": False, "error": error}

            source_obj = Path(source)
            dest_obj = Path(destination)

            if dest_obj.exists() and not overwrite:
                return {"success": False, "error": f"Destination exists: {destination}"}

            # Create parent directories if needed
            dest_obj.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(source_obj), str(dest_obj))

            return {
                "success": True,
                "source": str(source_obj),
                "destination": str(dest_obj),
                "moved": True
            }

        except Exception as e:
            return {"success": False, "error": f"Move error: {str(e)}"}

    @classmethod
    def copy_file(cls, source: str, destination: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Copy a file or directory.

        Args:
            source: Source path
            destination: Destination path
            recursive: Copy recursively for directories

        Returns:
            Result dictionary
        """
        try:
            # Validate paths
            is_valid, error = cls.validate_path(source, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            is_valid, error = cls.validate_path(destination, must_exist=False)
            if not is_valid:
                return {"success": False, "error": error}

            source_obj = Path(source)
            dest_obj = Path(destination)

            # Create parent directories if needed
            dest_obj.parent.mkdir(parents=True, exist_ok=True)

            if source_obj.is_dir():
                if recursive:
                    shutil.copytree(str(source_obj), str(dest_obj))
                else:
                    return {"success": False, "error": "Use recursive=True for directories"}
            else:
                shutil.copy2(str(source_obj), str(dest_obj))

            return {
                "success": True,
                "source": str(source_obj),
                "destination": str(dest_obj),
                "copied": True
            }

        except Exception as e:
            return {"success": False, "error": f"Copy error: {str(e)}"}

    @classmethod
    def search_files(cls, path: str, pattern: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Search for files by pattern.

        Args:
            path: Root directory
            pattern: Search pattern
            recursive: Search recursively

        Returns:
            Result dictionary with matches
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            if not path_obj.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            matches = []

            if recursive:
                for item in path_obj.rglob(pattern):
                    matches.append(str(item.relative_to(path_obj)))
            else:
                for item in path_obj.glob(pattern):
                    matches.append(str(item.relative_to(path_obj)))

            return {
                "success": True,
                "path": str(path_obj),
                "pattern": pattern,
                "matches": sorted(matches),
                "count": len(matches)
            }

        except Exception as e:
            return {"success": False, "error": f"Search error: {str(e)}"}

    @classmethod
    def get_file_info(cls, path: str) -> Dict[str, Any]:
        """
        Get file/directory metadata.

        Args:
            path: File/directory path

        Returns:
            Result dictionary with metadata
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)
            stat = path_obj.stat()

            info = {
                "success": True,
                "path": str(path_obj),
                "name": path_obj.name,
                "is_file": path_obj.is_file(),
                "is_directory": path_obj.is_dir(),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            }

            # Add file hash for files
            if path_obj.is_file():
                with open(path_obj, "rb") as f:
                    info["md5"] = hashlib.md5(f.read()).hexdigest()

            return info

        except Exception as e:
            return {"success": False, "error": f"Info error: {str(e)}"}

    @classmethod
    def file_exists(cls, path: str) -> Dict[str, Any]:
        """
        Check if path exists.

        Args:
            path: Path to check

        Returns:
            Result dictionary with exists boolean
        """
        try:
            path_obj = Path(path)
            return {
                "success": True,
                "path": str(path_obj),
                "exists": path_obj.exists(),
                "is_file": path_obj.is_file() if path_obj.exists() else False,
                "is_directory": path_obj.is_dir() if path_obj.exists() else False
            }
        except Exception as e:
            return {"success": False, "error": f"Check error: {str(e)}"}

    @classmethod
    def search_in_files(
        cls,
        path: str,
        query: str,
        pattern: str = "*",
        case_sensitive: bool = False,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Search for text within files.

        Args:
            path: Directory to search
            query: Text to find
            pattern: File pattern filter
            case_sensitive: Case-sensitive search
            recursive: Search recursively

        Returns:
            Result dictionary with matches
        """
        try:
            # Validate path
            is_valid, error = cls.validate_path(path, must_exist=True)
            if not is_valid:
                return {"success": False, "error": error}

            path_obj = Path(path)

            if not path_obj.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            matches = []

            # Find matching files
            if recursive:
                files = path_obj.rglob(pattern)
            else:
                files = path_obj.glob(pattern)

            for file_path in files:
                if not file_path.is_file():
                    continue

                try:
                    # Check file size
                    if file_path.stat().st_size > cls.MAX_FILE_SIZE:
                        continue

                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    # Search in content
                    if case_sensitive:
                        found = query in content
                    else:
                        found = query.lower() in content.lower()

                    if found:
                        # Get matching lines
                        matching_lines = []
                        for i, line in enumerate(content.splitlines(), 1):
                            if case_sensitive:
                                if query in line:
                                    matching_lines.append({
                                        "line": i,
                                        "content": line.strip()[:200]
                                    })
                            else:
                                if query.lower() in line.lower():
                                    matching_lines.append({
                                        "line": i,
                                        "content": line.strip()[:200]
                                    })

                        matches.append({
                            "file": str(file_path.relative_to(path_obj)),
                            "matches": matching_lines[:10],  # Limit matches
                            "total_matches": len(matching_lines)
                        })

                except Exception as e:
                    debug.log(f"Error reading {file_path}: {e}")
                    continue

            return {
                "success": True,
                "path": str(path_obj),
                "query": query,
                "pattern": pattern,
                "matches": matches,
                "files_with_matches": len(matches)
            }

        except Exception as e:
            return {"success": False, "error": f"Search error: {str(e)}"}


# Convenience functions for direct use

def get_filesystem_tools() -> List[Dict[str, Any]]:
    """Get filesystem tool definitions"""
    return FileSystemTools.get_tools()


def execute_filesystem_tool(
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a filesystem tool by name.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments

    Returns:
        Tool execution result
    """
    tool_map = {
        "read_file": FileSystemTools.read_file,
        "write_file": FileSystemTools.write_file,
        "list_directory": FileSystemTools.list_directory,
        "create_directory": FileSystemTools.create_directory,
        "delete_file": FileSystemTools.delete_file,
        "delete_directory": FileSystemTools.delete_directory,
        "move_file": FileSystemTools.move_file,
        "copy_file": FileSystemTools.copy_file,
        "search_files": FileSystemTools.search_files,
        "get_file_info": FileSystemTools.get_file_info,
        "file_exists": FileSystemTools.file_exists,
        "search_in_files": FileSystemTools.search_in_files,
    }

    if tool_name not in tool_map:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

    try:
        return tool_map[tool_name](**arguments)
    except Exception as e:
        return {"success": False, "error": f"Execution error: {str(e)}"}

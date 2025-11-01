# gpt4free MCP Server - Complete Usage Guide

## Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Available Tools](#available-tools)
- [Integration Examples](#integration-examples)
- [Troubleshooting](#troubleshooting)

## Introduction

The gpt4free MCP (Model Context Protocol) server enables AI assistants like Claude to access powerful capabilities:
- **Web Search**: Real-time web search using DuckDuckGo
- **Web Scraping**: Extract and clean text content from any web page
- **Image Generation**: Create images from text descriptions using various AI models

## Quick Start

### 1. Installation

Make sure gpt4free is installed with all dependencies:

```bash
# Install with all features
pip install -U g4f[all]

# Or install from source
git clone https://github.com/xtekky/gpt4free.git
cd gpt4free
pip install -e .
```

### 2. Start the MCP Server

```bash
# Using g4f command
g4f mcp

# Or using Python module
python -m g4f.mcp

# With debug logging
g4f mcp --debug
```

The server will:
- Listen on stdin for JSON-RPC requests
- Write responses to stdout
- Write debug/error messages to stderr

### 3. Test the Server

```bash
# Send a test request
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python -m g4f.mcp
```

Expected output:
```json
{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "serverInfo": {...}}}
```

## Configuration

### Claude Desktop

1. Locate your config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add the MCP server:

```json
{
  "mcpServers": {
    "gpt4free": {
      "command": "python",
      "args": ["-m", "g4f.mcp"],
      "description": "gpt4free MCP server with web search, scraping, and image generation"
    }
  }
}
```

3. Restart Claude Desktop

4. Verify in Claude: Ask "What tools do you have access to?" and you should see the gpt4free tools listed.

### VS Code with Cline Extension

Add to your Cline MCP settings:

```json
{
  "mcpServers": {
    "gpt4free": {
      "command": "python",
      "args": ["-m", "g4f.mcp"],
      "disabled": false
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can use the server. The command is:
```bash
python -m g4f.mcp
```

## Available Tools

### 1. web_search

Search the web for current information.

**Parameters:**
- `query` (string, required): Search query
- `max_results` (integer, optional): Maximum results to return (default: 5)

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "web_search",
    "arguments": {
      "query": "latest Python 3.12 features",
      "max_results": 5
    }
  }
}
```

**Example Usage in Claude:**
> "Search the web for the latest Python 3.12 features"

### 2. web_scrape

Extract text content from web pages.

**Parameters:**
- `url` (string, required): URL to scrape
- `max_words` (integer, optional): Maximum words to extract (default: 1000)

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "web_scrape",
    "arguments": {
      "url": "https://python.org",
      "max_words": 500
    }
  }
}
```

**Example Usage in Claude:**
> "Scrape the content from https://python.org and summarize it"

### 3. image_generation

Generate images from text descriptions.

**Parameters:**
- `prompt` (string, required): Image description
- `model` (string, optional): Image model (default: "flux")
- `width` (integer, optional): Width in pixels (default: 1024)
- `height` (integer, optional): Height in pixels (default: 1024)

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "image_generation",
    "arguments": {
      "prompt": "A serene mountain landscape at sunset",
      "width": 1024,
      "height": 1024
    }
  }
}
```

**Example Usage in Claude:**
> "Generate an image of a serene mountain landscape at sunset"

## Integration Examples

### Python Script

```python
import asyncio
import json
from g4f.mcp.server import MCPServer, MCPRequest

async def search_web(query: str):
    server = MCPServer()
    request = MCPRequest(
        jsonrpc="2.0",
        id=1,
        method="tools/call",
        params={
            "name": "web_search",
            "arguments": {"query": query}
        }
    )
    response = await server.handle_request(request)
    return response.result

# Run it
result = asyncio.run(search_web("Python tutorials"))
print(result)
```

### Command Line Testing

```bash
# Test initialize
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | g4f mcp

# Test list tools
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | g4f mcp

# Test web search
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"test"}}}' | g4f mcp
```

### Using with Shell Scripts

```bash
#!/bin/bash
# search.sh - Simple web search wrapper

query="$1"
request=$(cat <<EOF
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"$query"}}}
EOF
)

echo "$request" | python -m g4f.mcp | jq '.result.content[0].text | fromjson'
```

## Troubleshooting

### Server Won't Start

**Problem**: Server exits immediately or shows import errors

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install with all extras
pip install -U g4f[all]
```

### Tools Return Errors

**Problem**: Tools return error messages about missing packages

**Solution**: Install specific dependencies:
```bash
# For web search
pip install ddgs beautifulsoup4

# For web scraping
pip install aiohttp beautifulsoup4

# For image generation
pip install pillow
```

### Network Errors

**Problem**: Tools fail with connection errors

**Solution**:
- Check internet connectivity
- Some providers may be rate-limited
- Try different providers for image generation
- Check firewall settings

### Claude Desktop Not Finding Server

**Problem**: Claude doesn't show gpt4free tools

**Solution**:
1. Verify config file location and syntax
2. Check that Python is in PATH
3. Try absolute path to Python:
   ```json
   {
     "mcpServers": {
       "gpt4free": {
         "command": "/usr/bin/python3",
         "args": ["-m", "g4f.mcp"]
       }
     }
   }
   ```
4. Restart Claude Desktop completely
5. Check Claude logs for errors

### Debug Mode

Enable debug output:
```bash
# Redirect stderr to see debug messages
g4f mcp 2> mcp_debug.log

# Run with verbose output
g4f mcp --debug 2>&1 | tee mcp_output.log
```

### Verify Installation

Run the test script:
```bash
python etc/testing/test_mcp_server.py
```

Or the interactive demo:
```bash
python etc/testing/test_mcp_interactive.py
```

## Protocol Details

The MCP server implements JSON-RPC 2.0 over stdio transport.

**Supported Methods:**
- `initialize` - Initialize the connection
- `tools/list` - List all available tools  
- `tools/call` - Execute a tool
- `ping` - Health check

**Message Format:**
- Requests: One JSON object per line on stdin
- Responses: One JSON object per line on stdout
- Logs: Messages on stderr

## Advanced Usage

### Custom Tool Development

To add custom tools, see `g4f/mcp/tools.py`:

```python
from g4f.mcp.tools import MCPTool

class MyCustomTool(MCPTool):
    @property
    def description(self) -> str:
        return "My custom tool description"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        # Your implementation
        pass
```

Register in `g4f/mcp/server.py`:
```python
self.tools['my_tool'] = MyCustomTool()
```

## Support

- Documentation: [g4f/mcp/README.md](README.md)
- Issues: https://github.com/xtekky/gpt4free/issues
- MCP Specification: https://modelcontextprotocol.io/

## License

Part of the gpt4free project, licensed under GNU General Public License v3.0.

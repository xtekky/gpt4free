# gpt4free MCP Server

A Model Context Protocol (MCP) server implementation for gpt4free that provides AI assistants with access to web search, scraping, and image generation capabilities.

## Overview

The gpt4free MCP server exposes three main tools:

1. **Web Search** - Search the web using DuckDuckGo
2. **Web Scraping** - Extract and clean text content from web pages
3. **Image Generation** - Generate images from text prompts using various AI providers

## Installation

The MCP server is included with gpt4free. No additional installation is required beyond the base gpt4free package.

```bash
pip install -e .
```

## Usage

### Running the MCP Server

**Stdio Mode (Default)**

Start the MCP server using:

```bash
python -m g4f.mcp
```

Or using the g4f command:

```bash
g4f mcp
```

The server communicates over stdin/stdout using JSON-RPC 2.0 protocol.

**HTTP Mode**

Start the MCP server with HTTP transport:

```bash
g4f mcp --http --port 8765
```

This starts an HTTP server with the following endpoints:
- `POST http://localhost:8765/mcp` - MCP JSON-RPC endpoint
- `GET http://localhost:8765/health` - Health check endpoint

HTTP mode is useful for:
- Web-based integrations
- Testing with curl or HTTP clients
- Remote access (configure host with `--host`)

Options:
- `--http`: Enable HTTP transport instead of stdio
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--port PORT`: Port to bind to (default: 8765)

### Configuration for AI Assistants

**For Claude Desktop (Stdio)** - `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gpt4free": {
      "command": "python",
      "args": ["-m", "g4f.mcp"]
    }
  }
}
```

**For HTTP-based clients**:

Make POST requests to `http://localhost:8765/mcp` with JSON-RPC payloads.

Example with curl:
```bash
curl -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

**For VS Code with Cline**:

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

## Available Tools

### web_search

Search the web for information.

**Parameters:**
- `query` (string, required): The search query
- `max_results` (integer, optional): Maximum number of results (default: 5)

**Example:**
```json
{
  "name": "web_search",
  "arguments": {
    "query": "latest AI developments 2024",
    "max_results": 5
  }
}
```

### web_scrape

Scrape and extract text content from a web page.

**Parameters:**
- `url` (string, required): The URL to scrape
- `max_words` (integer, optional): Maximum words to extract (default: 1000)

**Example:**
```json
{
  "name": "web_scrape",
  "arguments": {
    "url": "https://example.com/article",
    "max_words": 1000
  }
}
```

### image_generation

Generate images from text prompts.

**Parameters:**
- `prompt` (string, required): Description of the image to generate
- `model` (string, optional): Image model to use (default: "flux")
- `width` (integer, optional): Image width in pixels (default: 1024)
- `height` (integer, optional): Image height in pixels (default: 1024)

**Example:**
```json
{
  "name": "image_generation",
  "arguments": {
    "prompt": "A serene mountain landscape at sunset",
    "width": 1024,
    "height": 1024
  }
}
```

## Protocol Details

The MCP server implements the Model Context Protocol using JSON-RPC 2.0 over stdio transport.

### Supported Methods

- `initialize` - Initialize connection with the server
- `tools/list` - List all available tools
- `tools/call` - Execute a tool with given arguments
- `ping` - Health check

### Example Request/Response

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "web_search",
    "arguments": {
      "query": "Python programming tutorials",
      "max_results": 3
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"query\": \"Python programming tutorials\", \"results\": [...], \"count\": 3}"
      }
    ]
  }
}
```

## Requirements

The MCP server requires the following dependencies (included in gpt4free):

- `aiohttp` - For async HTTP requests
- `beautifulsoup4` - For web scraping
- `ddgs` - For web search

These are automatically installed with:

```bash
pip install -r requirements.txt
```

## Error Handling

The server returns standard JSON-RPC error responses:

- `-32601`: Method not found
- `-32602`: Invalid parameters
- `-32603`: Internal error

Errors specific to tools are returned in the result object with an `error` field.

## Development

### Project Structure

```
g4f/mcp/
├── __init__.py      # Package initialization
├── __main__.py      # CLI entry point
├── server.py        # MCP server implementation
├── tools.py         # Tool implementations
└── README.md        # This file
```

### Adding New Tools

To add a new tool:

1. Create a new class inheriting from `MCPTool` in `tools.py`
2. Implement the required properties and methods
3. Register the tool in `MCPServer.__init__()` in `server.py`

Example:

```python
class MyNewTool(MCPTool):
    @property
    def description(self) -> str:
        return "Description of what the tool does"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param1"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        # Implementation
        pass
```

## Troubleshooting

### Server Won't Start

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Tools Return Errors

Check that:
- Network connectivity is available for web search and scraping
- URLs are valid and accessible
- Image generation providers are not rate-limited

### Debug Mode

The server writes diagnostic information to stderr. To see debug output:
```bash
python -m g4f.mcp 2> debug.log
```

## License

This MCP server is part of the gpt4free project and is licensed under the GNU General Public License v3.0.

## Contributing

Contributions are welcome! Please see the main gpt4free repository for contribution guidelines.

## Related Links

- [gpt4free Repository](https://github.com/xtekky/gpt4free)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP Documentation](https://modelcontextprotocol.io/docs)

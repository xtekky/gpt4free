#!/usr/bin/env python
"""
Example: Using the MCP Server Tools

This script demonstrates how to interact with the MCP server tools programmatically.
It shows how each tool can be used and what kind of results to expect.
"""

import asyncio
import json
from g4f.mcp.server import MCPServer, MCPRequest


async def demo_web_search():
    """Demonstrate web search tool"""
    print("\n" + "=" * 70)
    print("DEMO: Web Search Tool")
    print("=" * 70)
    
    server = MCPServer()
    
    # Create a tool call request for web search
    request = MCPRequest(
        jsonrpc="2.0",
        id=1,
        method="tools/call",
        params={
            "name": "web_search",
            "arguments": {
                "query": "Python programming tutorials",
                "max_results": 3
            }
        }
    )
    
    print("\nRequest:")
    print(json.dumps({
        "method": "tools/call",
        "params": request.params
    }, indent=2))
    
    print("\nExecuting web search...")
    response = await server.handle_request(request)
    
    if response.result:
        print("\nSuccess! Response:")
        content = response.result.get("content", [])
        if content:
            result_text = content[0].get("text", "")
            result_data = json.loads(result_text)
            print(json.dumps(result_data, indent=2))
    elif response.error:
        print(f"\nError: {response.error}")


async def demo_web_scrape():
    """Demonstrate web scraping tool"""
    print("\n" + "=" * 70)
    print("DEMO: Web Scrape Tool")
    print("=" * 70)
    
    server = MCPServer()
    
    # Create a tool call request for web scraping
    request = MCPRequest(
        jsonrpc="2.0",
        id=2,
        method="tools/call",
        params={
            "name": "web_scrape",
            "arguments": {
                "url": "https://example.com",
                "max_words": 200
            }
        }
    )
    
    print("\nRequest:")
    print(json.dumps({
        "method": "tools/call",
        "params": request.params
    }, indent=2))
    
    print("\nExecuting web scrape...")
    response = await server.handle_request(request)
    
    if response.result:
        print("\nSuccess! Response:")
        content = response.result.get("content", [])
        if content:
            result_text = content[0].get("text", "")
            result_data = json.loads(result_text)
            print(json.dumps(result_data, indent=2))
    elif response.error:
        print(f"\nError: {response.error}")


async def demo_image_generation():
    """Demonstrate image generation tool"""
    print("\n" + "=" * 70)
    print("DEMO: Image Generation Tool")
    print("=" * 70)
    
    server = MCPServer()
    
    # Create a tool call request for image generation
    request = MCPRequest(
        jsonrpc="2.0",
        id=3,
        method="tools/call",
        params={
            "name": "image_generation",
            "arguments": {
                "prompt": "A beautiful sunset over mountains",
                "model": "flux",
                "width": 512,
                "height": 512
            }
        }
    )
    
    print("\nRequest:")
    print(json.dumps({
        "method": "tools/call",
        "params": request.params
    }, indent=2))
    
    print("\nExecuting image generation...")
    response = await server.handle_request(request)
    
    if response.result:
        print("\nSuccess! Response:")
        content = response.result.get("content", [])
        if content:
            result_text = content[0].get("text", "")
            result_data = json.loads(result_text)
            # Don't print the full base64 image data, just show metadata
            if "image" in result_data and result_data["image"].startswith("data:"):
                result_data["image"] = result_data["image"][:100] + "... (base64 data truncated)"
            print(json.dumps(result_data, indent=2))
    elif response.error:
        print(f"\nError: {response.error}")


async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("gpt4free MCP Server - Tool Demonstrations")
    print("=" * 70)
    print("\nThis script demonstrates the three main tools available in the MCP server:")
    print("1. Web Search - Search the web using DuckDuckGo")
    print("2. Web Scrape - Extract content from web pages")
    print("3. Image Generation - Generate images from text prompts")
    print("\nNote: These tools require network access and may fail in isolated environments.")
    
    # Show tool information
    print("\n" + "=" * 70)
    print("Available Tools")
    print("=" * 70)
    
    server = MCPServer()
    for name, tool in server.tools.items():
        print(f"\n• {name}")
        print(f"  Description: {tool.description}")
        schema = tool.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        print(f"  Required parameters: {', '.join(required)}")
        print(f"  Optional parameters: {', '.join([k for k in properties.keys() if k not in required])}")
    
    # Run demos (these may fail without network access or required packages)
    try:
        await demo_web_search()
    except Exception as e:
        print(f"\n⚠ Web search demo failed: {e}")
        print("This is expected without network access or required packages (ddgs, beautifulsoup4)")
    
    try:
        await demo_web_scrape()
    except Exception as e:
        print(f"\n⚠ Web scrape demo failed: {e}")
        print("This is expected without network access or required packages (aiohttp, beautifulsoup4)")
    
    try:
        await demo_image_generation()
    except Exception as e:
        print(f"\n⚠ Image generation demo failed: {e}")
        print("This is expected without network access or image generation providers")
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print("\nTo use these tools in production:")
    print("1. Start the MCP server: g4f mcp")
    print("2. Configure your AI assistant to connect to it")
    print("3. The assistant can then use these tools to enhance its capabilities")
    print("\nSee g4f/mcp/README.md for detailed configuration instructions.")


if __name__ == "__main__":
    asyncio.run(main())

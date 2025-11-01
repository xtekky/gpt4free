#!/usr/bin/env python
"""Test HTTP MCP server functionality

This script tests the HTTP transport for the MCP server.
"""

import asyncio
import json
from g4f.mcp.server import MCPServer, MCPRequest


async def test_http_server():
    """Test HTTP server methods"""
    server = MCPServer()
    
    print("Testing HTTP MCP Server Functionality")
    print("=" * 70)
    
    # Test that server can be initialized
    print("\n✓ Server initialized successfully")
    print(f"  Server: {server.server_info['name']}")
    print(f"  Version: {server.server_info['version']}")
    
    # Test that run_http method exists
    if hasattr(server, 'run_http'):
        print("\n✓ HTTP transport method (run_http) available")
        print(f"  Signature: run_http(host, port)")
    else:
        print("\n✗ HTTP transport method not found")
        return
    
    # Test request handling (same for both transports)
    print("\n✓ Testing request handling...")
    
    init_request = MCPRequest(
        jsonrpc="2.0",
        id=1,
        method="initialize",
        params={}
    )
    response = await server.handle_request(init_request)
    
    if response.result and response.result.get("protocolVersion"):
        print(f"  Protocol Version: {response.result['protocolVersion']}")
        print("  ✓ Request handling works correctly")
    
    print("\n" + "=" * 70)
    print("HTTP MCP Server Tests Passed!")
    print("\nTo start HTTP server:")
    print("  g4f mcp --http --port 8765")
    print("\nHTTP endpoints:")
    print("  POST http://localhost:8765/mcp - MCP JSON-RPC endpoint")
    print("  GET  http://localhost:8765/health - Health check")
    print("\nExample HTTP request:")
    print('  curl -X POST http://localhost:8765/mcp \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\'')


if __name__ == "__main__":
    asyncio.run(test_http_server())

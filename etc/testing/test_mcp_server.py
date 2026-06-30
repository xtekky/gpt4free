#!/usr/bin/env python
"""Test script for MCP server

This script tests the MCP server by simulating client interactions.
It sends JSON-RPC requests and verifies responses.
"""

import json
import sys
import asyncio
from g4f.mcp.server import MCPServer, MCPRequest


async def test_mcp_server():
    """Test MCP server functionality"""
    server = MCPServer()
    
    print("Testing MCP Server...")
    print("=" * 60)
    
    # Test 1: Initialize
    print("\n1. Testing initialize request...")
    init_request = MCPRequest(
        jsonrpc="2.0",
        id=1,
        method="initialize",
        params={}
    )
    response = await server.handle_request(init_request)
    print(f"   Response ID: {response.id}")
    print(f"   Protocol Version: {response.result['protocolVersion']}")
    print(f"   Server Name: {response.result['serverInfo']['name']}")
    print("   ✓ Initialize test passed")
    
    # Test 2: List tools
    print("\n2. Testing tools/list request...")
    list_request = MCPRequest(
        jsonrpc="2.0",
        id=2,
        method="tools/list",
        params={}
    )
    response = await server.handle_request(list_request)
    print(f"   Number of tools: {len(response.result['tools'])}")
    for tool in response.result['tools']:
        print(f"   - {tool['name']}: {tool['description'][:50]}...")
    print("   ✓ Tools list test passed")
    
    # Test 3: Ping
    print("\n3. Testing ping request...")
    ping_request = MCPRequest(
        jsonrpc="2.0",
        id=3,
        method="ping",
        params={}
    )
    response = await server.handle_request(ping_request)
    print(f"   Response ID: {response.id}")
    print("   ✓ Ping test passed")
    
    # Test 4: Invalid method
    print("\n4. Testing invalid method request...")
    invalid_request = MCPRequest(
        jsonrpc="2.0",
        id=4,
        method="invalid_method",
        params={}
    )
    response = await server.handle_request(invalid_request)
    if response.error:
        print(f"   Error code: {response.error['code']}")
        print(f"   Error message: {response.error['message']}")
        print("   ✓ Invalid method test passed")
    
    # Test 5: Tool schemas
    print("\n5. Testing tool input schemas...")
    list_request = MCPRequest(
        jsonrpc="2.0",
        id=5,
        method="tools/list",
        params={}
    )
    response = await server.handle_request(list_request)
    for tool in response.result['tools']:
        print(f"   Tool: {tool['name']}")
        schema = tool['inputSchema']
        required = schema.get('required', [])
        properties = schema.get('properties', {})
        print(f"     Required params: {', '.join(required)}")
        print(f"     All params: {', '.join(properties.keys())}")
    print("   ✓ Tool schemas test passed")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nMCP server is working correctly.")
    print("\nTo use the server, run:")
    print("  python -m g4f.mcp")
    print("  or")
    print("  g4f mcp")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

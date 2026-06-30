#!/usr/bin/env python
"""Interactive MCP server test

This script simulates a client sending requests to the MCP server
and demonstrates how the tools work.
"""

import json
import sys
import asyncio
from io import StringIO


async def simulate_mcp_client():
    """Simulate an MCP client interacting with the server"""
    
    print("MCP Server Interactive Test")
    print("=" * 70)
    print("\nThis test simulates JSON-RPC 2.0 messages between client and server.")
    print("The MCP server uses stdio transport for communication.\n")
    
    from g4f.mcp.server import MCPServer, MCPRequest
    server = MCPServer()
    
    # Test sequence of requests
    test_requests = [
        {
            "name": "Initialize Connection",
            "request": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
        },
        {
            "name": "List Available Tools",
            "request": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
        },
        {
            "name": "Ping Server",
            "request": {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "ping",
                "params": {}
            }
        },
    ]
    
    for test in test_requests:
        print(f"\n{'─' * 70}")
        print(f"Test: {test['name']}")
        print(f"{'─' * 70}")
        
        # Show request
        print("\nClient Request:")
        print(json.dumps(test['request'], indent=2))
        
        # Create request object
        req_data = test['request']
        request = MCPRequest(
            jsonrpc=req_data.get("jsonrpc", "2.0"),
            id=req_data.get("id"),
            method=req_data.get("method"),
            params=req_data.get("params")
        )
        
        # Handle request
        response = await server.handle_request(request)
        
        # Show response
        print("\nServer Response:")
        response_dict = {
            "jsonrpc": response.jsonrpc,
            "id": response.id
        }
        if response.result is not None:
            response_dict["result"] = response.result
        if response.error is not None:
            response_dict["error"] = response.error
        
        print(json.dumps(response_dict, indent=2))
        
        await asyncio.sleep(0.1)  # Small delay between requests
    
    print(f"\n{'═' * 70}")
    print("Interactive Test Complete!")
    print(f"{'═' * 70}\n")
    
    print("Tool Descriptions:")
    print("-" * 70)
    for name, tool in server.tools.items():
        print(f"\n• {name}")
        print(f"  {tool.description}")
        schema = tool.input_schema
        if 'required' in schema:
            print(f"  Required: {', '.join(schema['required'])}")
        if 'properties' in schema:
            optional = [k for k in schema['properties'].keys() if k not in schema.get('required', [])]
            if optional:
                print(f"  Optional: {', '.join(optional)}")
    
    print(f"\n{'═' * 70}")
    print("How to Use the MCP Server:")
    print(f"{'═' * 70}\n")
    print("1. Start the server:")
    print("   $ python -m g4f.mcp")
    print("   or")
    print("   $ g4f mcp")
    print()
    print("2. Configure in Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):")
    print('   {')
    print('     "mcpServers": {')
    print('       "gpt4free": {')
    print('         "command": "python",')
    print('         "args": ["-m", "g4f.mcp"]')
    print('       }')
    print('     }')
    print('   }')
    print()
    print("3. Or test via stdin/stdout:")
    print('   $ echo \'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\' | python -m g4f.mcp')
    print()
    print("The server will:")
    print("  • Read JSON-RPC requests from stdin (one per line)")
    print("  • Process the request and execute tools if needed")
    print("  • Write JSON-RPC responses to stdout (one per line)")
    print("  • Write debug/error messages to stderr")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(simulate_mcp_client())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)

"""Android on-device actions exposed through the Python MCP registry.

The Android app hosts a native, loopback-only MCP server (see
``dev.g4f.android.mcp.MobileMcpServer``) that implements the seven MobileActions
tools from Google AI Edge Gallery.  This module registers matching proxies in
the regular gpt4free MCP server so chat/agent tool loops can call them like any
other MCP tool.

The proxies are only installed when the host app sets ``G4F_MOBILE_MCP=1``,
so desktop installs keep their previous tool list.  The endpoint can be
overridden with ``G4F_MOBILE_MCP_URL``.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Any, Dict

DEFAULT_MOBILE_MCP_URL = "http://127.0.0.1:1338/mcp"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def mobile_mcp_enabled() -> bool:
    """Return whether the native Android action server should be bridged."""
    return os.environ.get("G4F_MOBILE_MCP", "").strip().lower() in _TRUE_VALUES


def mobile_mcp_url() -> str:
    return os.environ.get("G4F_MOBILE_MCP_URL", DEFAULT_MOBILE_MCP_URL).strip()


def _string_schema(**properties: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": description}
            for name, description in properties.items()
        },
        "required": list(properties),
    }


_EMPTY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


class MobileActionTool:
    """Proxy for one action implemented by the Android app."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any],
                 parameters: tuple[str, ...]):
        self._name = name
        self._description = description
        self._input_schema = input_schema
        self._parameters = parameters

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            key: arguments[key]
            for key in self._parameters
            if key in arguments and arguments[key] is not None
        }
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": self._name, "arguments": payload},
        }).encode("utf-8")
        request = urllib.request.Request(
            mobile_mcp_url(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "MCP-Protocol-Version": "2025-06-18",
            },
            method="POST",
        )

        def send() -> Dict[str, Any]:
            # Loopback must never be routed through an HTTP proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, send)
        except Exception as exc:
            return {"error": f"Mobile action server unavailable: {exc}"}

        error = response.get("error")
        if error:
            return {"error": f"MCP error {error.get('code')}: {error.get('message')}"}
        result = response.get("result") or {}
        if result.get("isError"):
            structured = result.get("structuredContent")
            if isinstance(structured, dict) and structured.get("error"):
                return structured
            content = result.get("content") or []
            text = content[0].get("text") if content else "Mobile action failed"
            return {"error": text}
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        return result


def get_mobile_tools() -> Dict[str, MobileActionTool]:
    """Build the proxy tools; empty unless a host app enabled them."""
    if not mobile_mcp_enabled():
        return {}
    tools = (
        ("turnOnFlashlight", "Turns the flashlight on", _EMPTY_SCHEMA, ()),
        ("turnOffFlashlight", "Turns the flashlight off", _EMPTY_SCHEMA, ()),
        ("createContact", "Creates a contact in the phone's contact list.",
         _string_schema(
             firstName="The first name of the contact.",
             lastName="The last name of the contact.",
             phoneNumber="The phone number of the contact.",
             email="The email address of the contact.",
         ), ("firstName", "lastName", "phoneNumber", "email")),
        ("sendEmail", "Sends an email.",
         _string_schema(
             to="The email address of the recipient.",
             subject="The subject of the email.",
             body="The body of the email.",
         ), ("to", "subject", "body")),
        ("showLocationOnMap", "Shows a location on the map.",
         _string_schema(
             location="The location to search for. May be the name of a place, "
                      "a business, or an address.",
         ), ("location",)),
        ("openWifiSettings", "Opens the WiFi settings.", _EMPTY_SCHEMA, ()),
        ("createCalendarEvent", "Creates a new calendar event.",
         _string_schema(
             datetime="The date and time of the event in the format YYYY-MM-DDTHH:MM:SS.",
             title="The title of the event.",
         ), ("datetime", "title")),
    )
    return {
        name: MobileActionTool(name, description, schema, parameters)
        for name, description, schema, parameters in tools
    }

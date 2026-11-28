from __future__ import annotations

import unittest

from g4f.Provider.needs_auth.Antigravity import (
    AntigravityProvider,
    _sanitize_schema,
    build_antigravity_payload,
    ensure_roles_in_contents,
    gemini_to_antigravity,
    normalize_antigravity_contents,
)


def _function_call_names(contents):
    names = []
    for content in contents:
        for part in content.get("parts", []):
            if isinstance(part.get("functionCall"), dict):
                names.append(part["functionCall"].get("name"))
    return names


def _function_response_names(contents):
    names = []
    for content in contents:
        for part in content.get("parts", []):
            if isinstance(part.get("functionResponse"), dict):
                names.append(part["functionResponse"].get("name"))
    return names


class TestMessagesToGeminiFormat(unittest.TestCase):
    def test_tool_response_uses_function_name_instead_of_tool_call_id(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"q": "value"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        contents = AntigravityProvider._messages_to_gemini_format(
            messages, media=None
        )

        function_call = contents[1]["parts"][0]["functionCall"]
        self.assertEqual(function_call["name"], "lookup")
        self.assertEqual(function_call["id"], "call_1")
        self.assertEqual(function_call["args"], {"q": "value"})

        function_response = contents[2]["parts"][0]["functionResponse"]
        self.assertEqual(function_response["name"], "lookup")
        self.assertEqual(function_response["id"], "call_1")
        self.assertEqual(function_response["response"], {"result": "result"})

    def test_tool_call_id_in_name_is_ignored(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "call_1",
                "content": "result",
            },
        ]

        contents = AntigravityProvider._messages_to_gemini_format(
            messages, media=None
        )

        function_response = contents[2]["parts"][0]["functionResponse"]
        self.assertEqual(function_response["name"], "lookup")
        self.assertEqual(function_response["id"], "call_1")

    def test_tool_response_falls_back_to_message_name(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {
                "role": "tool",
                "name": "lookup",
                "tool_call_id": "call_1",
                "content": "result",
            },
        ]

        contents = AntigravityProvider._messages_to_gemini_format(
            messages, media=None
        )

        function_response = contents[1]["parts"][0]["functionResponse"]
        self.assertEqual(function_response["name"], "lookup")

    def test_malformed_tool_arguments_do_not_raise(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "not json"},
                    }
                ],
            },
        ]

        contents = AntigravityProvider._messages_to_gemini_format(
            messages, media=None
        )

        self.assertEqual(
            contents[1]["parts"][0]["functionCall"]["args"],
            {"input": "not json"},
        )


class TestNormalizeAntigravityContents(unittest.TestCase):
    def test_history_starting_with_function_call_gets_user_turn(self):
        contents = normalize_antigravity_contents(
            [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "id": "call_1",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "id": "call_1",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                },
            ]
        )

        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(contents[2]["role"], "user")
        self.assertEqual(_function_call_names(contents), ["lookup"])
        self.assertEqual(_function_response_names(contents), ["lookup"])

    def test_consecutive_model_turns_are_merged(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "thinking"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "id": "call_1",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "id": "call_1",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                },
            ]
        )

        roles = [content["role"] for content in contents]
        self.assertEqual(roles, ["user", "model", "user"])
        self.assertEqual(len(contents[1]["parts"]), 2)
        self.assertEqual(_function_call_names(contents), ["lookup"])

    def test_function_response_name_is_synced_with_matched_call(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": "look it up"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "id": "call_1",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "call_1",
                                "id": "call_1",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                },
            ]
        )

        function_response = contents[2]["parts"][0]["functionResponse"]
        self.assertEqual(function_response["name"], "lookup")
        self.assertEqual(function_response["id"], "call_1")

    def test_orphan_function_response_becomes_text(self):
        contents = normalize_antigravity_contents(
            [
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                }
            ]
        )

        self.assertEqual(_function_response_names(contents), [])
        self.assertIn("Function lookup returned: ok", contents[0]["parts"][0]["text"])

    def test_unanswered_function_call_becomes_text(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": "first"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "args": {"q": "value"},
                            }
                        }
                    ],
                },
                {"role": "user", "parts": [{"text": "second"}]},
            ]
        )

        self.assertEqual(_function_call_names(contents), [])
        model_text = contents[1]["parts"][0]["text"]
        self.assertIn("lookup", model_text)

    def test_trailing_function_call_becomes_text(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": "first"}]},
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "lookup", "args": {}}}],
                },
            ]
        )

        self.assertEqual(_function_call_names(contents), [])
        self.assertIn("lookup", contents[1]["parts"][0]["text"])
        self.assertEqual(contents[-1]["role"], "user")

    def test_trailing_model_turn_gets_user_turn(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "Hello there!"}]},
            ]
        )

        self.assertEqual(contents[-1]["role"], "user")

    def test_empty_history_gets_user_turn(self):
        contents = normalize_antigravity_contents([])

        self.assertEqual(contents, [{"role": "user", "parts": [{"text": "."}]}])

    def test_response_only_matches_immediately_preceding_call(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": "first"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "old", "id": "c1", "args": {}}}
                    ],
                },
                {"role": "user", "parts": [{"text": "never mind"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "new", "id": "c2", "args": {}}}
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "old",
                                "id": "c1",
                                "response": {"result": "stale"},
                            }
                        },
                        {
                            "functionResponse": {
                                "name": "new",
                                "id": "c2",
                                "response": {"result": "fresh"},
                            }
                        },
                    ],
                },
            ]
        )

        self.assertEqual(_function_call_names(contents), ["new"])
        self.assertEqual(_function_response_names(contents), ["new"])

    def test_empty_text_parts_do_not_create_empty_turns(self):
        contents = normalize_antigravity_contents(
            [
                {"role": "user", "parts": [{"text": ""}]},
                {"role": "model", "parts": [{"text": "answer"}]},
            ]
        )

        self.assertEqual(
            [content["role"] for content in contents],
            ["user", "model", "user"],
        )


class TestSanitizeSchema(unittest.TestCase):
    def test_array_without_items_gets_default_items(self):
        schema = {
            "type": "object",
            "properties": {"issues_found": {"type": "array"}},
        }

        sanitized = _sanitize_schema(schema)

        self.assertEqual(
            sanitized["properties"]["issues_found"]["items"],
            {"type": "string"},
        )

    def test_uppercase_array_without_items_gets_default_items(self):
        sanitized = _sanitize_schema({"type": "ARRAY"})

        self.assertEqual(sanitized["items"], {"type": "string"})

    def test_required_only_references_declared_properties(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path", "missing"],
        }

        sanitized = _sanitize_schema(schema)

        self.assertEqual(sanitized["required"], ["path"])

    def test_required_without_properties_is_removed(self):
        sanitized = _sanitize_schema({"type": "object", "required": ["missing"]})

        self.assertNotIn("required", sanitized)

    def test_union_type_is_reduced_and_marked_nullable(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": ["string", "null"]}},
        }

        sanitized = _sanitize_schema(schema)

        query = sanitized["properties"]["query"]
        self.assertEqual(query["type"], "string")
        self.assertTrue(query["nullable"])

    def test_unsupported_keywords_are_removed_recursively(self):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {
                    "type": "string",
                    "$ref": "#/definitions/value",
                    "const": "fixed",
                    "default": "x",
                },
                "choice": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                },
            },
        }

        sanitized = _sanitize_schema(schema)

        self.assertNotIn("$schema", sanitized)
        self.assertNotIn("additionalProperties", sanitized)
        value = sanitized["properties"]["value"]
        for key in ("$ref", "const", "default"):
            self.assertNotIn(key, value)
        self.assertNotIn("anyOf", sanitized["properties"]["choice"])

    def test_properties_without_type_becomes_object(self):
        sanitized = _sanitize_schema({"properties": {"a": {"type": "string"}}})

        self.assertEqual(sanitized["type"], "object")

    def test_non_list_required_is_removed(self):
        sanitized = _sanitize_schema(
            {"type": "object", "properties": {}, "required": True}
        )

        self.assertNotIn("required", sanitized)

    def test_boolean_property_schema_becomes_empty_schema(self):
        sanitized = _sanitize_schema(
            {"type": "object", "properties": {"free": True}}
        )

        self.assertEqual(sanitized["properties"]["free"], {})

    def test_enum_values_are_stringified(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "enum": [1, 2, 3]},
                "flag": {"type": "boolean", "enum": [True, False]},
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
        }

        sanitized = _sanitize_schema(schema)

        properties = sanitized["properties"]
        self.assertEqual(properties["count"]["enum"], ["1", "2", "3"])
        self.assertEqual(properties["flag"]["enum"], ["true", "false"])
        self.assertEqual(properties["mode"]["enum"], ["a", "b"])

    def test_tuple_items_use_first_schema(self):
        sanitized = _sanitize_schema(
            {
                "type": "array",
                "items": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
            }
        )

        self.assertEqual(sanitized["items"], {"type": "string"})

    def test_gemini_payload_sanitizes_tool_declarations(self):
        payload = gemini_to_antigravity(
            "gemini-3-flash",
            {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "name": "lookup",
                                    "description": "",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "items": {"type": "array"}
                                        },
                                        "required": ["items", "missing"],
                                    },
                                }
                            ]
                        }
                    ],
                }
            },
            "project-id",
        )

        declaration = payload["request"]["tools"][0][
            "functionDeclarations"
        ][0]
        parameters = declaration["parameters"]
        self.assertEqual(parameters["properties"]["items"]["items"], {"type": "string"})
        self.assertEqual(parameters["required"], ["items"])


class TestEnsureRolesInContents(unittest.TestCase):
    def test_long_tool_history_is_repaired_end_to_end(self):
        request_body = {
            "contents": [
                {
                    "role": "model",
                    "parts": [{"text": "I will look that up"}],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "id": "call_1",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "id": "call_1",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                },
            ]
        }

        processed = ensure_roles_in_contents(request_body, "gemini-3-flash")
        contents = processed["contents"]

        self.assertEqual(
            [content["role"] for content in contents], ["user", "model", "user"]
        )
        self.assertEqual(_function_call_names(contents), ["lookup"])
        self.assertEqual(_function_response_names(contents), ["lookup"])

    def test_build_payload_with_leading_tool_call(self):
        request_body = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "id": "call_1",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "id": "call_1",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                },
            ]
        }

        payload, selected_model, _ = build_antigravity_payload(
            "gemini-3-flash", request_body, "project-id"
        )

        self.assertEqual(selected_model, "gemini-3-flash")
        self.assertIn("contents", payload["request"])
        self.assertEqual(payload["request"]["contents"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()

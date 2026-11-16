"""Tool declarations must be constructible by every provider client, not just one.

Google GenAI types ``Schema.enum`` as ``list[str]``. A JSON-Schema-legal integer
enum therefore fails inside that client while the request is still being built:
nothing is sent, every provider on the chain fails the same way, and the whole
conversation class stops answering. These tests keep that class of schema out of
the stored tool definitions.
"""

import json
import unittest

from companion_v2.execution.models import encode
from companion_v2.model_gateway.models import ToolSpec, portable_tool_schema


def has_non_string_enum(node):
    """Report the first non-string enum found anywhere below ``node``."""
    if isinstance(node, dict):
        values = node.get("enum")
        if isinstance(values, list) and any(not isinstance(value, str) for value in values):
            return values
        for child in node.values():
            found = has_non_string_enum(child)
            if found is not None:
                return found
    elif isinstance(node, list):
        for child in node:
            found = has_non_string_enum(child)
            if found is not None:
                return found
    return None


class PortableToolSchemaTests(unittest.TestCase):
    def test_integer_enum_is_dropped_and_its_values_move_into_the_description(self):
        cleaned = portable_tool_schema({
            "type": "object",
            "properties": {
                "row": {"type": "integer", "enum": [0, 1, 2]},
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["row"], "additionalProperties": False,
        })

        self.assertEqual(cleaned["properties"]["row"], {"type": "integer", "description": "可选值：0、1、2"})
        # A string enum is exactly what every client can construct, so it stays.
        self.assertEqual(cleaned["properties"]["mode"], {"type": "string", "enum": ["a", "b"]})
        self.assertIsNone(has_non_string_enum(cleaned))

    def test_cleaning_reaches_properties_and_items(self):
        source = {
            "type": "object",
            "properties": {
                "pick": {"type": "array", "items": {"type": "integer", "enum": [1, 2]}},
                "count": {"type": "integer", "minimum": 1},
            },
        }

        cleaned = portable_tool_schema(source)

        self.assertIsNone(has_non_string_enum(cleaned))
        self.assertEqual(cleaned["properties"]["pick"]["items"]["description"], "可选值：1、2")
        self.assertEqual(cleaned["properties"]["count"], {"type": "integer", "minimum": 1})
        # The cleaner works on a copy: the stored business schema is not rewritten.
        self.assertEqual(source["properties"]["pick"]["items"], {"type": "integer", "enum": [1, 2]})

    def test_a_top_level_combinator_is_hoisted_into_the_description(self):
        """Anthropic refuses a tool whose ``input_schema`` starts with anyOf/oneOf/allOf.

        The requirement it carried must survive as prompt material, and the
        declaration must no longer contain the keyword, or every Claude-backed
        conversation fails while the request is still being built (issue #708).
        """
        cleaned = portable_tool_schema({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
        })

        for keyword in ("anyOf", "oneOf", "allOf"):
            self.assertNotIn(keyword, cleaned)
        self.assertEqual(cleaned["type"], "object")
        self.assertIn("同时提供 a", cleaned["description"])
        self.assertIn("同时提供 b", cleaned["description"])
        self.assertIn("；或 ", cleaned["description"])

    def test_a_top_level_anyOf_without_a_declared_type_still_becomes_an_object(self):
        cleaned = portable_tool_schema({"anyOf": [{"required": ["a"]}, {"required": ["b"]}]})

        self.assertNotIn("anyOf", cleaned)
        self.assertEqual(cleaned["type"], "object")

    def test_nested_combinators_are_cleaned_and_kept(self):
        """Only the root is rewritten: a nested combinator is legal for every client."""
        cleaned = portable_tool_schema({
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "string"}, {"type": "integer", "enum": [3, 4]}]},
            },
        })

        branches = cleaned["properties"]["value"]["anyOf"]
        self.assertEqual(len(branches), 2)
        self.assertEqual(branches[1], {"type": "integer", "description": "可选值：3、4"})
        self.assertIsNone(has_non_string_enum(cleaned))

    def test_a_combinator_free_root_schema_keeps_its_description(self):
        cleaned = portable_tool_schema({
            "type": "object",
            "description": "Already portable.",
            "properties": {"mode": {"type": "string", "enum": ["a"]}},
            "required": ["mode"], "additionalProperties": False,
        })

        self.assertEqual(cleaned["description"], "Already portable.")

    def test_cleaning_an_already_portable_schema_changes_nothing(self):
        clean = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["a"]}},
                 "required": ["mode"], "additionalProperties": False}

        self.assertEqual(portable_tool_schema(clean), clean)
        self.assertEqual(portable_tool_schema(portable_tool_schema(clean)), clean)

    def test_stored_tool_specs_are_cleaned_when_they_are_declared(self):
        spec = ToolSpec("game.move", "Move.", encode({
            "type": "object",
            "properties": {"row": {"type": "integer", "enum": [0, 1, 2]}},
            "required": ["row"], "additionalProperties": False,
        }))

        schema = json.loads(spec.parameters_json)
        self.assertEqual(schema["properties"]["row"], {"type": "integer", "description": "可选值：0、1、2"})

    def test_a_portable_schema_is_stored_byte_for_byte(self):
        clean = encode({"type": "object", "properties": {"mode": {"type": "string", "enum": ["a"]}},
                        "required": ["mode"], "additionalProperties": False})

        self.assertEqual(ToolSpec("device.read", "Read.", clean).parameters_json, clean)


if __name__ == "__main__":
    unittest.main()

"""Tests for g4f.tools.optimize_request."""

import unittest

from g4f.tools.optimize_request import optimize_tools


WORKFLOW = (
    "CRITICAL workflow:\n"
    "1. Plan tasks by writing {word} list with specific, actionable items\n"
    "2. Mark ONE {word} as in-progress before starting work\n"
    "3. Complete the work for that specific {word}\n"
    "4. Mark that {word} as completed IMMEDIATELY\n"
    "5. Move to next {word} and repeat"
)


class ManageTodoWorkflowTests(unittest.TestCase):
    def _optimize(self, word, prefix="CRITICAL "):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "manage_todo_list",
                    "description": prefix + WORKFLOW.format(word=word),
                },
            }
        ]
        optimized, _, _ = optimize_tools(tools)
        return optimized[0]["function"]["description"]

    def test_condenses_workflow_wording_variants(self):
        for word in ("todo", "to-do", "to do"):
            with self.subTest(word=word):
                description = self._optimize(word)
                self.assertEqual(
                    description,
                    "Workflow: write todos → mark one as in-progress → "
                    "complete it → mark completed → repeat.",
                )

    def test_condenses_workflow_without_prefix_and_uppercase(self):
        description = self._optimize("TO-DO", prefix="")
        self.assertEqual(
            description,
            "Workflow: write todos → mark one as in-progress → "
            "complete it → mark completed → repeat.",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for async provider timeout cleanup."""

import asyncio
import unittest

from g4f.providers.base_provider import wait_for


class TestWaitFor(unittest.TestCase):
    def test_timeout_closes_wrapped_generator(self):
        closed = False

        async def response():
            nonlocal closed
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed = True

        async def run():
            with self.assertRaises(TimeoutError):
                async for _ in wait_for(response(), timeout=0.01):
                    pass
            self.assertTrue(closed)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
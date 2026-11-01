from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"


class TestSandboxImages(unittest.TestCase):
    def test_desktop_image_puts_opencode_on_the_path(self):
        """The menu installs OpenCode under ~/.opencode/bin, not ~/.local/bin."""
        text = (DOCKER / "Dockerfile-sandbox-desktop").read_text(encoding="utf-8")
        self.assertIn("/home/pn/.opencode/bin", text)

    def test_slim_image_mirrors_the_sandbox_build(self):
        """The slim image ships g4f-go, Debian Python and the Node.js copy."""
        text = (DOCKER / "Dockerfile-slim").read_text(encoding="utf-8")
        self.assertIn("AS g4f-go-builder", text)
        self.assertIn("COPY --from=g4f-go-builder /out/g4f-go /usr/local/bin/g4f-go", text)
        self.assertIn("ARG SANDBOX_BASE=debian:", text)
        self.assertIn("apt-get install -y --no-install-recommends \\\n    python3", text)
        self.assertIn("AS node-dist", text)
        self.assertIn("COPY --from=node-dist /usr/local/bin/node /usr/local/bin/node", text)
        self.assertIn("COPY --from=node-dist /usr/local/lib/node_modules /usr/local/lib/node_modules", text)


if __name__ == "__main__":
    unittest.main()

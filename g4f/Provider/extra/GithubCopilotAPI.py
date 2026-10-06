from __future__ import annotations

from ..template import OpenaiTemplate


class GithubCopilotAPI(OpenaiTemplate):
    label = "GitHub Copilot API"
    url = "https://github.com/copilot"
    login_url = "https://aider.chat/docs/llms/github.html"
    working = True
    base_url = "https://api.githubcopilot.com"
    needs_auth = True
    models_needs_auth = True

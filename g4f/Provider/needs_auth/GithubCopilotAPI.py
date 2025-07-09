from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class GithubCopilotAPI(OpenaiAPI):
    label = "GitHub Copilot API"
    url = "https://github.com/copilot"
    login_url = "https://aider.chat/docs/llms/github.html"
    working = True
    api_base = "https://api.githubcopilot.com"
    needs_auth = True


# SKILL.md

## Using gpt4free as an LLM Server for Bots (Clawbot/OpenClaw)

### Overview
This skill covers running gpt4free as a local LLM server with an OpenAI-compatible REST API, custom model routing (config.yaml), and integration with bots like Clawbot or OpenClaw.

### Best Practices
- Start the API server with: `python -m g4f --port 8080` (or use `g4f api --debug --port 8080`)
- Use the `/v1` endpoint for OpenAI-compatible requests (e.g., POST to `http://localhost:8080/v1/chat/completions`)
- Define custom model routes in `config.yaml` to aggregate/fallback across providers
- Place `config.yaml` in your cookies directory (e.g., `~/.g4f/cookies/config.yaml`)
- For Clawbot/OpenClaw, patch their config to point to your gpt4free server (see `patch-openclaw.py`)
- Test with: `g4f client "Hello" --model openclaw` or Python client

### Common Pitfalls
- Not starting the server before connecting bots
- Incorrect config.yaml path or syntax errors
- Missing required Python dependencies (install with `pip install -r requirements.txt`)
- Not exposing the correct port (default 8080)
- Forgetting to patch bot configs to use your local endpoint

### Workflow Steps
1. Install and set up gpt4free (see README)
2. Start the API server: `python -m g4f --port 8080`
3. (Optional) Create or edit `config.yaml` for custom model routing:
	 ```yaml
	 models:
		 - name: "openclaw"
			 providers:
				 - provider: "GeminiCLI"
					 model: "gemini-3-flash-preview"
					 condition: "quota.models.gemini-3-flash-preview.remainingFraction > 0 and error_count < 3"
				 - provider: "Antigravity"
					 model: "gemini-3-flash"
				 - provider: "PollinationsAI"
					 model: "openai"
	 ```
4. Patch your bot config (e.g., OpenClaw) to use `http://localhost:8080/v1` as the base URL (see `scripts/patch-openclaw.py`)
5. Start your bot and verify it connects to gpt4free
6. Monitor logs and test with the Python client or CLI

### References
- [README.md](../README.md)
- [docs/config-yaml-routing.md](../docs/config-yaml-routing.md)
- [scripts/patch-openclaw.py](../scripts/patch-openclaw.py)
- [scripts/setup-openclaw.sh](../scripts/setup-openclaw.sh)
- [g4f/client/__init__.py](../g4f/client/__init__.py)
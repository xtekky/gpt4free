import json, sys, os

config_file = os.path.expanduser("~/.openclaw/openclaw.json")
if not os.path.exists(config_file):
    print("OpenClaw config not found. Please onboard OpenClaw first with `openclaw onboard` command.")
    sys.exit(1)

provider = {
    "baseUrl": "http://localhost:8080/v1",
    "api": "openai-completions",
    "models": [
        {
            "id": "openclaw",
            "name": "Custom GPT4Free",
            "reasoning": True,
            "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 256000,
            "maxTokens": 8192,
        }
    ],
}

try:
    with open(config_file) as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}

cfg.setdefault("models", {})["providers"] = cfg.get("models", {}).get("providers", {})
cfg["models"]["providers"]["gpt4free"] = provider
cfg["models"]["providers"]["g4f-perplexity"] = {
    "baseUrl": "https://perplexity.g4f-dev.workers.dev",
    "api": "openai-completions",
    "models": [
        {
            "id": "turbo",
            "name": "perplexity"
        }
    ]
}
cfg["tools"] = cfg.get("tools", {})
cfg["tools"]["web"] = cfg["tools"].get("web", {})
cfg["tools"]["web"]["search"] = {
    "provider": "g4f-perplexity",
}

with open(config_file, "w") as f:
    json.dump(cfg, f, indent=2)

print("OpenClaw config patched via Python.")
print("You can now run `openclaw models set gpt4free/openclaw` to apply the new model config and `openclaw gateway restart` to restart the gateway with the new config.")
"""Configuration management for G4F Agent."""

import os
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# --- FIX START ---
# To break a circular import cycle, we duplicate the path logic from the client.
# This ensures the agent uses the same base directory without causing an import error.
def get_base_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    else:  # Linux and other UNIX-like
        return Path.home() / ".config"

# Define the main config directory, consistent with the client
G4F_CLI_CONFIG_DIR = get_base_config_dir() / "g4f-cli"
# --- FIX END ---


class ApprovalMode(Enum):
    SUGGEST = "suggest"
    AUTO_EDIT = "auto-edit"
    FULL_AUTO = "full-auto"

@dataclass
class Config:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    approval_mode: ApprovalMode = ApprovalMode.SUGGEST
    sandbox_enabled: bool = True
    git_integration: bool = True
    max_file_size: int = 1_000_000
    debug_mode: bool = False
    theme: str = "monokai"

# Define a dedicated subdirectory for the agent within the main config folder
CONFIG_DIR = G4F_CLI_CONFIG_DIR / "agent"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
CONFIG_DIR.mkdir(exist_ok=True)

def load_config() -> Config:
    config_data = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except: pass
    
    if os.getenv("G4F_AGENT_MODEL"): config_data["model"] = os.getenv("G4F_AGENT_MODEL")
    if os.getenv("G4F_AGENT_MODE"): config_data["approval_mode"] = os.getenv("G4F_AGENT_MODE")

    if "approval_mode" in config_data:
        try: config_data["approval_mode"] = ApprovalMode(config_data["approval_mode"])
        except: config_data.pop("approval_mode")
        
    return Config(**config_data)

def save_config(config: Config):
    config_dict = {k: v.value if isinstance(v, Enum) else v for k, v in config.__dict__.items()}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, indent=2)

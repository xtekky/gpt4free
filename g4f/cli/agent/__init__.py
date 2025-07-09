"""G4F Agent - AI-powered coding assistant for your terminal."""

__version__ = "2.0.0"

from .agent import EnhancedAgent
from .config import Config, ApprovalMode, load_config, save_config
from .actions import Action, ActionType

__all__ = [ "EnhancedAgent", "Config", "ApprovalMode", "Action", "ActionType", "load_config", "save_config" ]

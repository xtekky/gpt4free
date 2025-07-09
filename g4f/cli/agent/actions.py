"""Action types and definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ActionType(Enum):
    """Enumeration of all possible actions the agent can perform."""
    # File System
    CREATE = "create"
    EDIT = "edit"
    DELETE = "delete"
    READ = "read"
    LIST = "list"
    
    # Code & Execution
    EXECUTE = "execute"
    SEARCH = "search"
    
    # Version Control
    GIT = "git"
    
    # Interaction & Planning
    CLARIFY = "clarify"
    WEB_SEARCH = "web_search"
    
    # Deprecated / Legacy
    VIEW = "view"


@dataclass
class Action:
    """Represents a single, atomic action for the agent to perform."""
    type: ActionType
    target: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None
    
    # For 'edit' action
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    # For 'execute' action
    method: Optional[str] = None
    
    # For 'clarify' action
    question: Optional[str] = None

    # For 'web_search' action
    query: Optional[str] = None
    
    # Metadata
    risk_level: str = "low"

    def __post_init__(self):
        """Handle content aliasing for different action types for convenience."""
        if self.type == ActionType.CLARIFY and self.question:
            self.reason = self.question
        if self.type == ActionType.WEB_SEARCH and self.query:
            self.content = self.query

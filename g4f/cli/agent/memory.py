# g4f/cli/agent/memory.py

"""Agent memory and context management."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class AgentMemory:
    """Manages the agent's short-term (conversation) and long-term (project) memory."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.memory_dir = config_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)
        
        self.project_memory = self._load_project_memory()
        self.conversation_history: List[Dict[str, str]] = []

    def _get_new_memory_structure(self) -> Dict[str, Any]:
        """Returns a clean slate for project memory."""
        return {
            "created": datetime.now().isoformat(), "last_updated": datetime.now().isoformat(),
            "actions": [], "notes": [], "file_history": {}, "dependencies": [], "test_results": []
        }

    def _load_project_memory(self) -> Dict[str, Any]:
        """Loads project-specific memory from a JSON file."""
        memory_file = self.memory_dir / f"{Path.cwd().name}_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                pass
        return self._get_new_memory_structure()
    
    def save_project_memory(self) -> None:
        """Saves the current project memory to disk."""
        memory_file = self.memory_dir / f"{Path.cwd().name}_memory.json"
        self.project_memory["last_updated"] = datetime.now().isoformat()
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(self.project_memory, f, indent=2)
        except IOError:
            print(f"Warning: Could not save project memory to {memory_file}")

    def add_action(self, action) -> None:
        """Add an action to long-term memory."""
        action_dict = {
            "type": action.type.value, "target": action.target, "reason": action.reason,
            "timestamp": datetime.now().isoformat()
        }
        self.project_memory["actions"].append(action_dict)
        self.project_memory["actions"] = self.project_memory["actions"][-100:] # Prune
        self.save_project_memory()

    def add_note(self, note: str, category: str = "general"):
        self.project_memory["notes"].append({
            "timestamp": datetime.now().isoformat(), "category": category, "content": note
        })
        self.save_project_memory()
        
    def get_context(self) -> str:
        """Constructs a detailed string of the current project context for the LLM."""
        context_parts = []
        
        try:
            files = list(Path(".").rglob("*"))
            file_list = [str(p) for p in files if p.is_file() and not any(part.startswith('.') or part in ['node_modules', '__pycache__'] for part in p.parts)]
            if file_list:
                context_parts.append(f"**Project Files (first 50):**\n- " + "\n- ".join(file_list[:50]))
        except Exception:
            context_parts.append("Could not list project files.")

        recent_actions = self.project_memory.get("actions", [])[-5:]
        if recent_actions:
            action_summary = [f"{a.get('type', '?')} on {a.get('target', '?')}" for a in recent_actions]
            context_parts.append("**Recent Actions:**\n- " + "\n- ".join(action_summary))

        recent_notes = self.project_memory.get("notes", [])[-5:]
        if recent_notes:
            notes_summary = [f"[{n.get('category', 'note')}] {n.get('content', '')}" for n in recent_notes]
            context_parts.append("**Recent Notes:**\n- " + "\n- ".join(notes_summary))
        
        return "\n\n".join(context_parts) if context_parts else "No project context available yet."

    def add_conversation(self, role: str, content: str):
        """Adds a message to the short-term conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep the conversation history from growing too large, but ALWAYS keep the first user prompt.
        if len(self.conversation_history) > 10:
            # Find the first message from the user, which is the original goal.
            first_user_prompt = next((msg for msg in self.conversation_history if msg["role"] == "user"), None)
            
            # Get the most recent messages.
            recent_messages = self.conversation_history[-9:]
            
            # Reconstruct the history, ensuring the original prompt is always first.
            new_history = []
            if first_user_prompt:
                new_history.append(first_user_prompt)
                # Add recent messages, avoiding duplication of the first prompt if it's also recent.
                for msg in recent_messages:
                    if msg != first_user_prompt:
                        new_history.append(msg)
            else:
                # Fallback if no user prompt is found for some reason.
                new_history = self.conversation_history[-10:]
            
            self.conversation_history = new_history


    def get_conversation_context(self) -> str:
        """Returns a string representation of the recent conversation history for the LLM."""
        if not self.conversation_history:
            return "No conversation history yet."
        
        output = []
        for msg in self.conversation_history:
            role = msg['role'].capitalize()
            content = msg['content']
            if role.lower() == 'observation':
                output.append(f"**System Observation:**\n{content}")
            else:
                output.append(f"**{role}:** {content[:1500]}") # Increase context length
        return "\n".join(output)
    
    def clear_memory(self, confirm: bool = False):
        """Clears the project memory file if confirmed."""
        if not confirm:
            print("Confirmation not provided. Memory not cleared.")
            return
        self.conversation_history = []
        self.project_memory = self._get_new_memory_structure()
        self.save_project_memory()
        print("Project memory has been cleared.")

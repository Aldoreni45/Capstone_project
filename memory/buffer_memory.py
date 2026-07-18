from typing import List, Dict
from custom_logging.logger import app_logger

class BufferMemory:
    """Stores full conversation history without pruning or summarization."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Adds a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        app_logger.debug(f"Added message to BufferMemory: role={role}")

    def get_history(self) -> str:
        """Formats the history as a continuous text block."""
        formatted = []
        for msg in self.messages:
            role_name = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_name}: {msg['content']}")
        return "\n".join(formatted)

    def get_messages(self) -> List[Dict[str, str]]:
        """Returns the raw message list."""
        return self.messages

    def clear(self):
        """Clears all conversation history."""
        self.messages.clear()
        app_logger.info("BufferMemory cleared.")

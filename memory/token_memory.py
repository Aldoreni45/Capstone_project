from typing import List, Dict
from custom_logging.logger import app_logger

class TokenWindowMemory:
    """Stores conversation history up to a maximum token threshold, pruning oldest messages first."""

    def __init__(self, max_tokens: int = 2000):
        self.messages: List[Dict[str, str]] = []
        self.max_tokens = max_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic calculation: average 1 word is roughly 1.3 tokens."""
        return int(len(text.split()) * 1.3)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._prune()

    def _prune(self):
        """Prunes older messages until total size is within max_tokens."""
        total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in self.messages)
        
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            app_logger.debug(f"Pruned message from history to fit token limit: {removed['role']}")
            total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in self.messages)

    def get_history(self) -> str:
        formatted = []
        for msg in self.messages:
            role_name = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_name}: {msg['content']}")
        return "\n".join(formatted)

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def clear(self):
        self.messages.clear()
        app_logger.info("TokenWindowMemory cleared.")

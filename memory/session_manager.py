from typing import Dict, Any, Union
from memory.buffer_memory import BufferMemory
from memory.token_memory import TokenWindowMemory
from memory.summary_memory import SummaryMemory
from custom_logging.logger import app_logger

# Types of memory available
MemoryType = Union[BufferMemory, TokenWindowMemory, SummaryMemory]

class SessionMemoryManager:
    """Manages conversational memories per session and memory type choice."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        # Maps session_id -> MemoryType
        self.sessions: Dict[str, MemoryType] = {}

    def get_session_memory(self, session_id: str, memory_type: str = "buffer") -> MemoryType:
        """
        Retrieves or initializes the memory instance for a session.
        Supported types: 'buffer', 'token', 'summary'
        """
        memory_type = memory_type.lower().strip()
        key = f"{session_id}_{memory_type}"
        
        if key not in self.sessions:
            app_logger.info(f"Initializing new memory structure '{memory_type}' for session '{session_id}'")
            if memory_type == "buffer":
                self.sessions[key] = BufferMemory()
            elif memory_type == "token":
                self.sessions[key] = TokenWindowMemory(max_tokens=2000)
            elif memory_type == "summary":
                self.sessions[key] = SummaryMemory(self.groq_api_key)
            else:
                app_logger.warning(f"Unknown memory type '{memory_type}'. Defaulting to buffer.")
                self.sessions[key] = BufferMemory()
                
        return self.sessions[key]

    def clear_session(self, session_id: str):
        """Clears all memory versions for a specific session."""
        keys_to_remove = [k for k in self.sessions.keys() if k.startswith(f"{session_id}_")]
        for k in keys_to_remove:
            self.sessions[k].clear()
            del self.sessions[k]
        app_logger.info(f"Cleared session history for session ID: {session_id}")

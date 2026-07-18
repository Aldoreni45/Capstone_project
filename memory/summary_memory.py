from typing import List, Dict
import httpx
from config.settings import settings
from custom_logging.logger import app_logger

class SummaryMemory:
    """Maintains a running LLM-generated summary of the conversation."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.default_model = settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")
        self.summary: str = ""
        self.buffer: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        # Update summary if we have at least 2 messages in buffer
        if len(self.buffer) >= 2:
            self._update_summary()

    def _update_summary(self):
        if not self.groq_api_key:
            # Fallback if no API key
            self.summary = "\n".join([f"{m['role']}: {m['content']}" for m in self.buffer])
            return

        conversation_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.buffer])
        prompt = f"""
Provide a concise, updated summary of the following conversation, integrating the previous summary if available.
Focus on research topics, key terms, questions asked, and answers provided.

Previous Summary:
"{self.summary}"

New messages to incorporate:
{conversation_str}

Updated Summary (keep it under 300 words):
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300
            }
            with httpx.Client() as client:
                response = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12.0)
                if response.status_code == 200:
                    self.summary = response.json()["choices"][0]["message"]["content"].strip()
                    app_logger.info("SummaryMemory updated successfully.")
                    self.buffer.clear()  # Clear buffer once summarized
                else:
                    app_logger.warning(f"Failed to update SummaryMemory: {response.text}")
        except Exception as e:
            app_logger.error(f"Error updating summary memory: {str(e)}")

    def get_history(self) -> str:
        """Returns the summary along with any remaining buffer messages."""
        history = [f"Conversation Summary: {self.summary}"]
        for msg in self.buffer:
            role_name = "User" if msg["role"] == "user" else "Assistant"
            history.append(f"{role_name}: {msg['content']}")
        return "\n".join(history)

    def clear(self):
        self.summary = ""
        self.buffer.clear()
        app_logger.info("SummaryMemory cleared.")

from typing import Optional, List, Any
import os
from groq import Groq as GroqClient
from groq.types.chat import ChatCompletionMessage


class GroqProvider:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = GroqClient(api_key=self.api_key)
        self.model = model

    def add_user_message(self, messages: list, message: Any):
        if isinstance(message, dict):
            messages.append(message)
        elif hasattr(message, "content"):
            messages.append({"role": "user", "content": message.content or ""})
        else:
            messages.append({"role": "user", "content": str(message)})

    def add_assistant_message(self, messages: list, message: Any):
        if isinstance(message, dict):
            messages.append(message)
            return

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if hasattr(message, "content") and message.content is not None:
            assistant_msg["content"] = message.content
        else:
            assistant_msg["content"] = None

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            assistant_msg["tool_calls"] = [
                tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

    def add_tool_messages(self, messages: list, tool_messages: list):
        for tool_msg in tool_messages:
            messages.append(tool_msg)

    def text_from_message(self, message: Any) -> str:
        if message is None:
            return ""
        if hasattr(message, "content"):
            return message.content or ""
        if isinstance(message, dict):
            return message.get("content") or ""
        return str(message)

    def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[list] = None,
        max_tokens: Optional[int] = 8000,
        **kwargs,
    ) -> ChatCompletionMessage:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_completion_tokens"] = max_tokens

        if stop_sequences:
            params["stop"] = stop_sequences

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message


# Alias for convenience
Groq = GroqProvider

from typing import Any, Optional
from core.groq import GroqProvider
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(
        self,
        groq_service: Optional[GroqProvider] = None,
        clients: Optional[dict[str, MCPClient]] = None,
        claude_service: Optional[Any] = None,
        **kwargs,
    ):
        self.groq_service: GroqProvider = groq_service or claude_service
        self.claude_service = self.groq_service  # Backward compatibility alias
        self.clients: dict[str, MCPClient] = clients or {}
        self.messages: list[dict[str, Any]] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            tools = await ToolManager.get_all_tools(self.clients)
            response = self.groq_service.chat(
                messages=self.messages,
                tools=tools,
            )

            self.groq_service.add_assistant_message(self.messages, response)

            if getattr(response, "tool_calls", None):
                content_text = self.groq_service.text_from_message(response)
                if content_text:
                    print(content_text)
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                self.groq_service.add_tool_messages(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.groq_service.text_from_message(
                    response
                )
                break

        return final_text_response

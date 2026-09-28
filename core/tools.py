import json
from typing import Optional, Literal, List, Any
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict[str, Any]]:
        """Gets all tools from the provided clients formatted for Groq function calling."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success"] | Literal["error"] = "success",
    ) -> dict[str, Any]:
        """Builds a tool result message dictionary for Groq."""
        return {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": text,
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message: Any
    ) -> List[dict[str, Any]]:
        """Executes a list of tool requests against the provided clients."""
        tool_requests = getattr(message, "tool_calls", None) or []
        if isinstance(message, dict):
            tool_requests = message.get("tool_calls", []) or []

        tool_result_blocks: list[dict[str, Any]] = []
        for tool_request in tool_requests:
            if hasattr(tool_request, "id"):
                tool_use_id = tool_request.id
                func = tool_request.function
                tool_name = func.name
                arguments = func.arguments
            elif isinstance(tool_request, dict):
                tool_use_id = tool_request.get("id", "")
                func = tool_request.get("function", {})
                tool_name = (
                    func.get("name", "")
                    if isinstance(func, dict)
                    else getattr(func, "name", "")
                )
                arguments = (
                    func.get("arguments", "{}")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", "{}")
                )
            else:
                continue

            if isinstance(arguments, str):
                try:
                    tool_input = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    tool_input = {}
            elif isinstance(arguments, dict):
                tool_input = arguments
            else:
                tool_input = {}

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id, "Could not find that tool", "error"
                )
                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(content_list)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    content_json,
                    "error"
                    if tool_output and tool_output.isError
                    else "success",
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    json.dumps({"error": error_message}),
                    "error",
                )

            tool_result_blocks.append(tool_result_part)
        return tool_result_blocks

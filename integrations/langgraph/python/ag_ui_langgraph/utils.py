import ast
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ag_ui.core import (
    AssistantMessage as AGUIAssistantMessage,
)
from ag_ui.core import (
    BinaryInputContent,
    ReasoningMessage,
    TextInputContent,
)
from ag_ui.core import (
    FunctionCall as AGUIFunctionCall,
)
from ag_ui.core import (
    Message as AGUIMessage,
)
from ag_ui.core import (
    SystemMessage as AGUISystemMessage,
)
from ag_ui.core import (
    ToolCall as AGUIToolCall,
)
from ag_ui.core import (
    ToolMessage as AGUIToolMessage,
)
from ag_ui.core import (
    UserMessage as AGUIUserMessage,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .types import LangGraphReasoning, SchemaKeys, State

DEFAULT_SCHEMA_KEYS = ["tools"]


def filter_object_by_schema_keys(
    obj: Dict[str, Any], schema_keys: List[str]
) -> Dict[str, Any]:
    if not obj:
        return {}
    return {k: v for k, v in obj.items() if k in schema_keys}


def get_stream_payload_input(
    *,
    mode: str,
    state: State,
    schema_keys: SchemaKeys,
) -> Union[State, None]:
    input_payload = state if mode == "start" else None
    if input_payload and schema_keys and schema_keys.get("input"):
        input_payload = filter_object_by_schema_keys(
            input_payload, [*DEFAULT_SCHEMA_KEYS, *schema_keys["input"]]
        )
    return input_payload


def stringify_if_needed(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    return json.dumps(item)


def _try_parse_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except Exception:
        return None


def _normalize_tool_result_data(raw_content: Any) -> Any:
    """
    Normalize tool output into a JSON-friendly value.

    - Preserves structured tool outputs (dict/list/etc)
    - If raw_content is a JSON string, parses it into structured data
    - Falls back to a JSON-safe string/value
    """
    safe = make_json_safe(raw_content)
    if isinstance(safe, str):
        stripped = safe.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            parsed = _try_parse_json(stripped)
            if parsed is not None:
                return parsed
            # Some tool stacks stringify python literals (single quotes, True/False).
            # Try to recover structured data safely.
            try:
                literal = ast.literal_eval(stripped)
                # Keep only structured container types / primitives.
                if (
                    isinstance(literal, (dict, list, tuple, str, int, float, bool))
                    or literal is None
                ):
                    return make_json_safe(literal)
            except Exception:
                pass
    return safe


def wrap_tool_result_content(
    *,
    tool_call_id: Optional[str],
    tool_name: Optional[str],
    raw_content: Any,
    ok: Optional[bool] = None,
) -> str:
    """
    Wrap tool results in a stable JSON envelope so clients can parse reliably.

    Envelope shape (v1):
      {
        "ok": bool,
        "tool": str | null,
        "toolCallId": str | null,
        "data": any,
        "meta": { "format": "ag_ui_tool_result_v1" }
      }

    If raw_content is already an envelope (dict or JSON string), it is normalized
    and returned without double-wrapping.
    """
    # Pass-through / normalize if already an envelope dict
    if isinstance(raw_content, dict) and (
        "data" in raw_content
        and ("toolCallId" in raw_content or "tool_call_id" in raw_content)
    ):
        normalized = dict(raw_content)
        normalized.setdefault("meta", {"format": "ag_ui_tool_result_v1"})
        if "toolCallId" not in normalized and "tool_call_id" in normalized:
            normalized["toolCallId"] = normalized.get("tool_call_id")
        if tool_call_id and not normalized.get("toolCallId"):
            normalized["toolCallId"] = tool_call_id
        if tool_name and normalized.get("tool") is None:
            normalized["tool"] = tool_name
        if ok is not None:
            normalized["ok"] = ok
        return json.dumps(normalized, default=json_safe_stringify)

    # If it's a JSON string that looks like an envelope, normalize and return
    if isinstance(raw_content, str):
        parsed = _try_parse_json(raw_content)
        if isinstance(parsed, dict) and (
            "data" in parsed and ("toolCallId" in parsed or "tool_call_id" in parsed)
        ):
            return wrap_tool_result_content(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                raw_content=parsed,
                ok=ok,
            )

    data = _normalize_tool_result_data(raw_content)
    if ok is None:
        ok = not (
            isinstance(data, dict)
            and ("error" in data or "errors" in data or "exception" in data)
        )

    envelope: Dict[str, Any] = {
        "ok": ok,
        "tool": tool_name,
        "toolCallId": tool_call_id,
        "data": data,
        "meta": {"format": "ag_ui_tool_result_v1"},
    }
    return json.dumps(envelope, default=json_safe_stringify)


def convert_langchain_multimodal_to_agui(
    content: List[Dict[str, Any]],
) -> List[Union[TextInputContent, BinaryInputContent]]:
    """Convert LangChain's multimodal content to AG-UI format."""
    agui_content = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                agui_content.append(
                    TextInputContent(type="text", text=item.get("text", ""))
                )
            elif item.get("type") == "image_url":
                image_url_data = item.get("image_url", {})
                url = (
                    image_url_data.get("url", "")
                    if isinstance(image_url_data, dict)
                    else image_url_data
                )

                # Parse data URLs to extract base64 data
                if url.startswith("data:"):
                    # Format: data:mime_type;base64,data
                    parts = url.split(",", 1)
                    header = parts[0]
                    data = parts[1] if len(parts) > 1 else ""
                    mime_type = (
                        header.split(":")[1].split(";")[0]
                        if ":" in header
                        else "image/png"
                    )

                    agui_content.append(
                        BinaryInputContent(
                            type="binary", mime_type=mime_type, data=data
                        )
                    )
                else:
                    # Regular URL or ID
                    agui_content.append(
                        BinaryInputContent(
                            type="binary",
                            mime_type="image/png",  # Default MIME type
                            url=url,
                        )
                    )
    return agui_content


def interleave_reasoning_messages(
    agui_messages: List[AGUIMessage],
    reasoning_messages: List[ReasoningMessage],
) -> List[AGUIMessage]:
    """
    Insert accumulated streaming reasoning messages before their corresponding assistant messages.

    This ensures reasoning is preserved in MESSAGES_SNAPSHOT even when the LLM provider
    doesn't persist reasoning content in the final AIMessage (e.g., Anthropic extended thinking).

    The function matches reasoning messages to assistant messages by order: the first
    accumulated reasoning goes before the first assistant message, etc.

    Args:
        agui_messages: Messages converted from LangChain (may already have reasoning from AIMessage content)
        reasoning_messages: Accumulated streaming reasoning messages

    Returns:
        Messages with streaming reasoning interleaved before assistant messages
    """
    if not reasoning_messages:
        return agui_messages

    result: List[AGUIMessage] = []
    reasoning_iter = iter(reasoning_messages)
    current_reasoning = next(reasoning_iter, None)

    for msg in agui_messages:
        # Skip reasoning messages that were already extracted from AIMessage content
        # (langchain_messages_to_agui already extracts reasoning from content blocks)
        if msg.role == "reasoning":
            result.append(msg)
            continue

        # Insert accumulated streaming reasoning before assistant message
        if msg.role == "assistant" and current_reasoning is not None:
            # Check if this reasoning isn't already in the result
            # (to avoid duplicates if langchain_messages_to_agui already extracted it)
            existing_ids = {m.id for m in result if m.role == "reasoning"}
            if current_reasoning.id not in existing_ids:
                result.append(current_reasoning)
            current_reasoning = next(reasoning_iter, None)

        result.append(msg)

    return result


def langchain_messages_to_agui(messages: List[BaseMessage]) -> List[AGUIMessage]:
    agui_messages: List[AGUIMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            # Handle multimodal content
            if isinstance(message.content, list):
                content = convert_langchain_multimodal_to_agui(message.content)
            else:
                content = stringify_if_needed(resolve_message_content(message.content))

            agui_messages.append(
                AGUIUserMessage(
                    id=str(message.id),
                    role="user",
                    content=content,
                    name=message.name,
                )
            )
        elif isinstance(message, AIMessage):
            # Extract reasoning blocks first (Anthropic: "thinking", OpenAI: "reasoning")
            # These appear BEFORE the assistant message in the conversation
            if isinstance(message.content, list):
                reasoning_index = 0
                for block in message.content:
                    if isinstance(block, dict):
                        # Anthropic uses "thinking", OpenAI uses "reasoning"
                        reasoning_text = block.get("thinking") or block.get("reasoning")
                        if reasoning_text:
                            agui_messages.append(
                                ReasoningMessage(
                                    id=f"{message.id}-reasoning-{reasoning_index}",
                                    role="reasoning",
                                    content=reasoning_text,
                                )
                            )
                            reasoning_index += 1

            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    AGUIToolCall(
                        id=str(tc["id"]),
                        type="function",
                        function=AGUIFunctionCall(
                            name=tc["name"],
                            arguments=json.dumps(tc.get("args", {})),
                        ),
                    )
                    for tc in message.tool_calls
                ]

            agui_messages.append(
                AGUIAssistantMessage(
                    id=str(message.id),
                    role="assistant",
                    content=stringify_if_needed(
                        resolve_message_content(message.content)
                    ),
                    tool_calls=tool_calls,
                    name=message.name,
                )
            )
        elif isinstance(message, SystemMessage):
            agui_messages.append(
                AGUISystemMessage(
                    id=str(message.id),
                    role="system",
                    content=stringify_if_needed(
                        resolve_message_content(message.content)
                    ),
                    name=message.name,
                )
            )
        elif isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None)
            agui_messages.append(
                AGUIToolMessage(
                    id=str(message.id),
                    role="tool",
                    content=wrap_tool_result_content(
                        tool_call_id=message.tool_call_id,
                        tool_name=tool_name,
                        raw_content=message.content,
                    ),
                    tool_call_id=message.tool_call_id,
                )
            )
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")
    return agui_messages


def convert_agui_multimodal_to_langchain(
    content: List[Union[TextInputContent, BinaryInputContent]],
) -> List[Dict[str, Any]]:
    """Convert AG-UI multimodal content to LangChain's multimodal format."""
    langchain_content = []
    for item in content:
        if isinstance(item, TextInputContent):
            langchain_content.append({"type": "text", "text": item.text})
        elif isinstance(item, BinaryInputContent):
            # LangChain uses image_url format (OpenAI-style)
            content_dict = {"type": "image_url"}

            # Prioritize url, then data, then id
            if item.url:
                content_dict["image_url"] = {"url": item.url}
            elif item.data:
                # Construct data URL from base64 data
                content_dict["image_url"] = {
                    "url": f"data:{item.mime_type};base64,{item.data}"
                }
            elif item.id:
                # Use id as a reference (some providers may support this)
                content_dict["image_url"] = {"url": item.id}

            langchain_content.append(content_dict)

    return langchain_content


def agui_messages_to_langchain(messages: List[AGUIMessage]) -> List[BaseMessage]:
    langchain_messages = []
    for message in messages:
        role = message.role
        if role == "user":
            # Handle multimodal content
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                content = convert_agui_multimodal_to_langchain(message.content)
            else:
                content = str(message.content)

            langchain_messages.append(
                HumanMessage(
                    id=message.id,
                    content=content,
                    name=message.name,
                )
            )
        elif role == "assistant":
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments)
                            if hasattr(tc, "function") and tc.function.arguments
                            else {},
                            "type": "tool_call",
                        }
                    )
            langchain_messages.append(
                AIMessage(
                    id=message.id,
                    content=message.content or "",
                    tool_calls=tool_calls,
                    name=message.name,
                )
            )
        elif role == "system":
            langchain_messages.append(
                SystemMessage(
                    id=message.id,
                    content=message.content,
                    name=message.name,
                )
            )
        elif role == "tool":
            langchain_messages.append(
                ToolMessage(
                    id=message.id,
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                )
            )
        elif role == "reasoning":
            # Skip reasoning messages when converting to LangChain
            # Reasoning is AG-UI specific for client display and should not be sent
            # back to LLM providers (Anthropic expects "thinking" not "reasoning",
            # and thinking content is typically not included in conversation history)
            continue
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return langchain_messages


def resolve_reasoning_content(chunk: Any) -> LangGraphReasoning | None:
    content = chunk.content
    if not content:
        return None

    # Anthropic reasoning response (LangChain ChatAnthropic returns "thinking" field)
    if isinstance(content, list) and content and content[0]:
        if not content[0].get("thinking"):
            return None
        return LangGraphReasoning(
            text=content[0]["thinking"], type="text", index=content[0].get("index", 0)
        )

    # OpenAI reasoning response
    if hasattr(chunk, "additional_kwargs"):
        reasoning = chunk.additional_kwargs.get("reasoning", {})
        summary = reasoning.get("summary", [])
        if summary:
            data = summary[0]
            if not data or not data.get("text"):
                return None
            return LangGraphReasoning(
                type="text", text=data["text"], index=data.get("index", 0)
            )

    return None


def resolve_message_content(content: Any) -> str | None:
    if not content:
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list) and content:
        content_text = next(
            (
                c.get("text")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ),
            None,
        )
        return content_text

    return None


def flatten_user_content(content: Any) -> str:
    """
    Flatten multimodal content into plain text.
    Used for backwards compatibility or when multimodal is not supported.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, TextInputContent):
                if item.text:
                    parts.append(item.text)
            elif isinstance(item, BinaryInputContent):
                # Add descriptive placeholder for binary content
                if item.filename:
                    parts.append(f"[Binary content: {item.filename}]")
                elif item.url:
                    parts.append(f"[Binary content: {item.url}]")
                else:
                    parts.append(f"[Binary content: {item.mime_type}]")
        return "\n".join(parts)

    return str(content)


def camel_to_snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def json_safe_stringify(o):
    if is_dataclass(o):  # dataclasses like Flight(...)
        return asdict(o)
    if hasattr(o, "model_dump"):  # pydantic v2
        return o.model_dump()
    if hasattr(o, "dict"):  # pydantic v1
        return o.dict()
    if hasattr(o, "__dict__"):  # plain objects
        return vars(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)  # last resort


def make_json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """
    Convert `value` into something that `json.dumps` can always handle.

    Rules (in order):
    - primitives → as-is
    - Enum → its .value (recursively made safe)
    - dict → keys & values made safe
    - list/tuple/set/frozenset → list of safe values
    - dataclasses → asdict() then recurse
    - Pydantic-style models → model_dump()/dict()/to_dict() then recurse
    - objects with __dict__ → vars(obj) then recurse
    - everything else → repr(obj)

    Cycles are detected and replaced with the string "<recursive>".
    """
    if _seen is None:
        _seen = set()

    obj_id = id(value)
    if obj_id in _seen:
        return "<recursive>"

    # --- 1. Primitives -----------------------------------------------------
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # --- 2. Enum → use underlying value -----------------------------------
    if isinstance(value, Enum):
        return make_json_safe(value.value, _seen)

    # --- 3. Dicts ----------------------------------------------------------
    if isinstance(value, dict):
        _seen.add(obj_id)
        return {
            make_json_safe(k, _seen): make_json_safe(v, _seen) for k, v in value.items()
        }

    # --- 4. Iterable containers -------------------------------------------
    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(obj_id)
        return [make_json_safe(v, _seen) for v in value]

    # --- 5. Dataclasses ----------------------------------------------------
    if is_dataclass(value):
        _seen.add(obj_id)
        return make_json_safe(asdict(value), _seen)

    # --- 6. Pydantic-like models (v2: model_dump) -------------------------
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        _seen.add(obj_id)
        try:
            return make_json_safe(value.model_dump(), _seen)
        except Exception:
            # fall through to other options
            pass

    # --- 7. Pydantic v1-style / other libs with .dict() -------------------
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        _seen.add(obj_id)
        try:
            return make_json_safe(value.dict(), _seen)
        except Exception:
            pass

    # --- 8. Generic "to_dict" pattern -------------------------------------
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        _seen.add(obj_id)
        try:
            return make_json_safe(value.to_dict(), _seen)
        except Exception:
            pass

    # --- 9. Generic Python objects with __dict__ --------------------------
    if hasattr(value, "__dict__"):
        _seen.add(obj_id)
        try:
            return make_json_safe(vars(value), _seen)
        except Exception:
            pass

    # --- 10. Last resort ---------------------------------------------------
    return repr(value)

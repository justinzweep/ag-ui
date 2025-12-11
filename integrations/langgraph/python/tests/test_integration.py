"""
Integration tests for LangGraph agent event handling.

These tests verify the full event flow from LangGraph events through to AG-UI events,
catching integration bugs like missing yield statements or incorrect event dispatching.
"""

import unittest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ag_ui.core import EventType


@dataclass
class MockResponseMetadata:
    """Mock response metadata for LangChain chunks."""

    finish_reason: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def create_tool_call_chunk(
    id: str = "tool-call-123",
    name: Optional[str] = None,
    args: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a tool call chunk dict (LangGraph uses dicts for tool_call_chunks)."""
    return {"id": id, "name": name, "args": args}


@dataclass
class MockChunk:
    """Mock LangChain message chunk for integration testing."""

    id: str = "chunk-123"
    content: Any = None
    response_metadata: MockResponseMetadata = field(default_factory=MockResponseMetadata)
    tool_call_chunks: List[Dict[str, Any]] = field(default_factory=list)
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockInterrupt:
    """Mock LangGraph interrupt."""

    value: Any = None


@dataclass
class MockTask:
    """Mock LangGraph task."""

    interrupts: List[MockInterrupt] = field(default_factory=list)


@dataclass
class MockAgentState:
    """Mock LangGraph agent state."""

    values: Dict[str, Any] = field(default_factory=lambda: {"messages": []})
    tasks: List[MockTask] = field(default_factory=list)
    next: tuple = ()


def create_langgraph_event(
    event_type: str,
    chunk: MockChunk,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a mock LangGraph event structure."""
    return {
        "event": event_type,
        "data": {"chunk": chunk},
        "metadata": metadata or {"emit-messages": True, "emit-tool-calls": True},
    }


def get_event_types(events: List[Any]) -> List[str]:
    """Extract event type values from a list of event objects."""
    types = []
    for e in events:
        if hasattr(e, "type"):
            # Handle both enum and string types
            event_type = e.type.value if hasattr(e.type, "value") else e.type
            types.append(event_type)
    return types


class TestReasoningIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for reasoning event flow through _handle_single_event."""

    def setUp(self):
        """Set up test fixtures."""
        from ag_ui_langgraph.agent import LangGraphAgent

        self.mock_graph = MagicMock()
        self.mock_graph.get_input_jsonschema.return_value = {"properties": {}}
        self.mock_graph.get_output_jsonschema.return_value = {"properties": {}}
        mock_config_schema = MagicMock()
        mock_config_schema.schema.return_value = {"properties": {}}
        self.mock_graph.config_schema.return_value = mock_config_schema

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)

        # Initialize active_run state as it would be during a real run
        self.agent.active_run = {
            "id": "run-123",
            "run_id": "run-123",
            "reasoning_process": None,
            "has_function_streaming": False,
            "reasoning_messages": [],
        }
        self.agent.messages_in_process = {}

    async def _collect_events(self, event: Dict[str, Any], state: Any = None) -> List[Any]:
        """Collect all events yielded by _handle_single_event."""
        events = []
        async for e in self.agent._handle_single_event(event, state or {}):
            events.append(e)
        return events

    async def test_thinking_content_emits_reasoning_events(self):
        """Full integration: LangGraph thinking content -> AG-UI REASONING_* events."""
        # Create a LangGraph event with Anthropic thinking content
        chunk = MockChunk(
            id="msg-123",
            content=[{"thinking": "Let me analyze this step by step...", "type": "thinking", "index": 0}],
        )
        event = create_langgraph_event("on_chat_model_stream", chunk)

        # Process the event
        events = await self._collect_events(event)

        # Should have emitted reasoning events
        self.assertGreater(len(events), 0, "No events were emitted for thinking content")

        # Verify event types
        event_types = get_event_types(events)

        # Should include REASONING_START
        self.assertIn(
            EventType.REASONING_START.value,
            event_types,
            f"REASONING_START not found in events: {event_types}",
        )

    async def test_thinking_content_emits_full_event_sequence(self):
        """Verify full reasoning event sequence: START -> MESSAGE_START -> CONTENT."""
        chunk = MockChunk(
            id="msg-123",
            content=[{"thinking": "First thought", "type": "thinking", "index": 0}],
        )
        event = create_langgraph_event("on_chat_model_stream", chunk)

        events = await self._collect_events(event)
        event_types = get_event_types(events)

        # Verify event sequence
        expected_sequence = [
            EventType.REASONING_START.value,
            EventType.REASONING_MESSAGE_START.value,
            EventType.REASONING_MESSAGE_CONTENT.value,
        ]

        for expected in expected_sequence:
            self.assertIn(
                expected,
                event_types,
                f"{expected} not found in event sequence: {event_types}",
            )

    async def test_multiple_thinking_chunks_emit_multiple_content_events(self):
        """Multiple thinking chunks should emit multiple REASONING_MESSAGE_CONTENT events."""
        # First chunk
        chunk1 = MockChunk(
            id="msg-123",
            content=[{"thinking": "First part...", "type": "thinking", "index": 0}],
        )
        event1 = create_langgraph_event("on_chat_model_stream", chunk1)

        events1 = await self._collect_events(event1)
        self.assertGreater(len(events1), 0)

        # Second chunk (same index continues the block)
        chunk2 = MockChunk(
            id="msg-123",
            content=[{"thinking": "Second part...", "type": "thinking", "index": 0}],
        )
        event2 = create_langgraph_event("on_chat_model_stream", chunk2)

        events2 = await self._collect_events(event2)
        self.assertGreater(len(events2), 0)

        # Verify second chunk emits content event
        event_types = get_event_types(events2)

        self.assertIn(
            EventType.REASONING_MESSAGE_CONTENT.value,
            event_types,
            "Second chunk should emit REASONING_MESSAGE_CONTENT",
        )

    async def test_reasoning_end_when_switching_to_text(self):
        """When thinking ends and text begins, should emit REASONING_END events."""
        # First: thinking content
        thinking_chunk = MockChunk(
            id="msg-123",
            content=[{"thinking": "Thinking...", "type": "thinking", "index": 0}],
        )
        thinking_event = create_langgraph_event("on_chat_model_stream", thinking_chunk)
        await self._collect_events(thinking_event)

        # Verify reasoning_process is set
        self.assertIsNotNone(self.agent.active_run.get("reasoning_process"))

        # Second: text content (no thinking)
        text_chunk = MockChunk(
            id="msg-123",
            content="Hello, here is the answer.",
        )
        text_event = create_langgraph_event("on_chat_model_stream", text_chunk)

        events = await self._collect_events(text_event)
        event_types = get_event_types(events)

        # Should include reasoning end events
        self.assertIn(
            EventType.REASONING_MESSAGE_END.value,
            event_types,
            f"REASONING_MESSAGE_END not found when switching to text: {event_types}",
        )
        self.assertIn(
            EventType.REASONING_END.value,
            event_types,
            f"REASONING_END not found when switching to text: {event_types}",
        )

    async def test_no_reasoning_events_for_regular_text(self):
        """Regular text content should NOT emit reasoning events."""
        chunk = MockChunk(
            id="msg-123",
            content="Just regular text without thinking",
        )
        event = create_langgraph_event("on_chat_model_stream", chunk)

        events = await self._collect_events(event)
        event_types = get_event_types(events)

        # Should NOT have reasoning events
        reasoning_events = [t for t in event_types if t and "REASONING" in t]
        self.assertEqual(
            len(reasoning_events),
            0,
            f"Regular text should not emit reasoning events: {event_types}",
        )


class TestToolCallIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for tool call event flow through _handle_single_event."""

    def setUp(self):
        """Set up test fixtures."""
        from ag_ui_langgraph.agent import LangGraphAgent

        self.mock_graph = MagicMock()
        self.mock_graph.get_input_jsonschema.return_value = {"properties": {}}
        self.mock_graph.get_output_jsonschema.return_value = {"properties": {}}
        mock_config_schema = MagicMock()
        mock_config_schema.schema.return_value = {"properties": {}}
        self.mock_graph.config_schema.return_value = mock_config_schema

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)
        self.agent.active_run = {
            "id": "run-123",
            "run_id": "run-123",
            "reasoning_process": None,
            "has_function_streaming": False,
        }
        self.agent.messages_in_process = {}

    async def _collect_events(self, event: Dict[str, Any], state: Any = None) -> List[Any]:
        """Collect all events yielded by _handle_single_event."""
        events = []
        async for e in self.agent._handle_single_event(event, state or {}):
            events.append(e)
        return events

    async def test_tool_call_start_event(self):
        """Tool call start should emit TOOL_CALL_START event."""
        tool_chunk = create_tool_call_chunk(id="call-123", name="get_weather", args=None)
        chunk = MockChunk(
            id="msg-123",
            content="",
            tool_call_chunks=[tool_chunk],
        )
        event = create_langgraph_event("on_chat_model_stream", chunk)

        events = await self._collect_events(event)

        self.assertGreater(len(events), 0, "No events emitted for tool call start")

        event_types = get_event_types(events)

        self.assertIn(
            EventType.TOOL_CALL_START.value,
            event_types,
            f"TOOL_CALL_START not found: {event_types}",
        )

    async def test_tool_call_args_event(self):
        """Tool call args should emit TOOL_CALL_ARGS event."""
        # First: start the tool call
        start_chunk = create_tool_call_chunk(id="call-123", name="get_weather", args=None)
        chunk1 = MockChunk(id="msg-123", content="", tool_call_chunks=[start_chunk])
        event1 = create_langgraph_event("on_chat_model_stream", chunk1)
        await self._collect_events(event1)

        # Second: send args
        args_chunk = create_tool_call_chunk(id="call-123", name=None, args='{"city": "NYC"}')
        chunk2 = MockChunk(id="msg-123", content="", tool_call_chunks=[args_chunk])
        event2 = create_langgraph_event("on_chat_model_stream", chunk2)

        events = await self._collect_events(event2)

        event_types = get_event_types(events)

        self.assertIn(
            EventType.TOOL_CALL_ARGS.value,
            event_types,
            f"TOOL_CALL_ARGS not found: {event_types}",
        )

    async def test_tool_call_end_event(self):
        """Tool call end should emit TOOL_CALL_END event."""
        # Start the tool call
        start_chunk = create_tool_call_chunk(id="call-123", name="get_weather", args=None)
        chunk1 = MockChunk(id="msg-123", content="", tool_call_chunks=[start_chunk])
        event1 = create_langgraph_event("on_chat_model_stream", chunk1)
        await self._collect_events(event1)

        # End the tool call (no tool_call_chunks)
        chunk2 = MockChunk(id="msg-123", content="", tool_call_chunks=[])
        event2 = create_langgraph_event("on_chat_model_stream", chunk2)

        events = await self._collect_events(event2)

        event_types = get_event_types(events)

        self.assertIn(
            EventType.TOOL_CALL_END.value,
            event_types,
            f"TOOL_CALL_END not found: {event_types}",
        )

    async def test_full_tool_call_sequence(self):
        """Full tool call flow: START -> ARGS -> END."""
        collected_types = []

        # Start
        start_chunk = create_tool_call_chunk(id="call-123", name="search", args=None)
        chunk1 = MockChunk(id="msg-123", content="", tool_call_chunks=[start_chunk])
        event1 = create_langgraph_event("on_chat_model_stream", chunk1)
        events1 = await self._collect_events(event1)
        collected_types.extend(get_event_types(events1))

        # Args
        args_chunk = create_tool_call_chunk(id="call-123", name=None, args='{"q": "test"}')
        chunk2 = MockChunk(id="msg-123", content="", tool_call_chunks=[args_chunk])
        event2 = create_langgraph_event("on_chat_model_stream", chunk2)
        events2 = await self._collect_events(event2)
        collected_types.extend(get_event_types(events2))

        # End
        chunk3 = MockChunk(id="msg-123", content="", tool_call_chunks=[])
        event3 = create_langgraph_event("on_chat_model_stream", chunk3)
        events3 = await self._collect_events(event3)
        collected_types.extend(get_event_types(events3))

        # Verify sequence
        self.assertIn(EventType.TOOL_CALL_START.value, collected_types)
        self.assertIn(EventType.TOOL_CALL_ARGS.value, collected_types)
        self.assertIn(EventType.TOOL_CALL_END.value, collected_types)


class TestTextMessageIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for text message event flow."""

    def setUp(self):
        """Set up test fixtures."""
        from ag_ui_langgraph.agent import LangGraphAgent

        self.mock_graph = MagicMock()
        self.mock_graph.get_input_jsonschema.return_value = {"properties": {}}
        self.mock_graph.get_output_jsonschema.return_value = {"properties": {}}
        mock_config_schema = MagicMock()
        mock_config_schema.schema.return_value = {"properties": {}}
        self.mock_graph.config_schema.return_value = mock_config_schema

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)
        self.agent.active_run = {
            "id": "run-123",
            "run_id": "run-123",
            "reasoning_process": None,
            "has_function_streaming": False,
        }
        self.agent.messages_in_process = {}

    async def _collect_events(self, event: Dict[str, Any], state: Any = None) -> List[Any]:
        """Collect all events yielded by _handle_single_event."""
        events = []
        async for e in self.agent._handle_single_event(event, state or {}):
            events.append(e)
        return events

    async def test_text_message_start_event(self):
        """First text chunk should emit TEXT_MESSAGE_START."""
        chunk = MockChunk(
            id="msg-123",
            content="Hello",
        )
        event = create_langgraph_event("on_chat_model_stream", chunk)

        events = await self._collect_events(event)

        self.assertGreater(len(events), 0, "No events emitted for text content")

        event_types = get_event_types(events)

        self.assertIn(
            EventType.TEXT_MESSAGE_START.value,
            event_types,
            f"TEXT_MESSAGE_START not found: {event_types}",
        )

    async def test_text_message_content_event(self):
        """Text content should emit TEXT_MESSAGE_CONTENT."""
        chunk = MockChunk(
            id="msg-123",
            content="Hello world",
        )
        event = create_langgraph_event("on_chat_model_stream", chunk)

        events = await self._collect_events(event)
        event_types = get_event_types(events)

        self.assertIn(
            EventType.TEXT_MESSAGE_CONTENT.value,
            event_types,
            f"TEXT_MESSAGE_CONTENT not found: {event_types}",
        )

    async def test_text_message_end_event(self):
        """Empty chunk after text should emit TEXT_MESSAGE_END."""
        # First: text content
        chunk1 = MockChunk(id="msg-123", content="Hello")
        event1 = create_langgraph_event("on_chat_model_stream", chunk1)
        await self._collect_events(event1)

        # Second: empty content (signals end)
        chunk2 = MockChunk(id="msg-123", content="")
        event2 = create_langgraph_event("on_chat_model_stream", chunk2)

        events = await self._collect_events(event2)
        event_types = get_event_types(events)

        self.assertIn(
            EventType.TEXT_MESSAGE_END.value,
            event_types,
            f"TEXT_MESSAGE_END not found: {event_types}",
        )


class TestInterruptIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for interrupt event flow."""

    def setUp(self):
        """Set up test fixtures."""
        from ag_ui_langgraph.agent import LangGraphAgent

        self.mock_graph = MagicMock()
        self.mock_graph.get_input_jsonschema.return_value = {"properties": {}}
        self.mock_graph.get_output_jsonschema.return_value = {"properties": {}}
        mock_config_schema = MagicMock()
        mock_config_schema.schema.return_value = {"properties": {}}
        self.mock_graph.config_schema.return_value = mock_config_schema
        self.mock_graph.aget_state = AsyncMock()

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)

        # Initialize active_run state
        self.agent.active_run = {
            "id": "run-123",
            "run_id": "run-123",
            "reasoning_process": None,
            "has_function_streaming": False,
            "mode": "start",
            "schema_keys": {"input": [], "output": [], "config": [], "context": []},
        }

    async def test_interrupt_detection_emits_run_finished_with_interrupt(self):
        """When interrupts are detected, should emit RunFinished with outcome='interrupt'."""
        # Mock agent state with interrupts
        interrupt = MockInterrupt(value={"reason": "human_approval", "tool": "send_email"})
        task = MockTask(interrupts=[interrupt])
        mock_state = MockAgentState(
            values={"messages": []},
            tasks=[task],
        )

        # Prepare input
        from ag_ui.core import RunAgentInput

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-123",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )

        # Mock config
        mock_config = {"configurable": {"thread_id": "thread-123"}}

        # Call prepare_stream which detects interrupts
        result = await self.agent.prepare_stream(input_data, mock_state, mock_config)

        # Should have events_to_dispatch with interrupt outcome
        self.assertIn("events_to_dispatch", result)
        events = result["events_to_dispatch"]

        # Find RunFinishedEvent
        run_finished = None
        for event in events:
            if hasattr(event, "type") and event.type == EventType.RUN_FINISHED:
                run_finished = event
                break

        self.assertIsNotNone(run_finished, "RunFinishedEvent not found in events")
        self.assertEqual(run_finished.outcome, "interrupt")
        self.assertIsNotNone(run_finished.interrupt)

    async def test_interrupt_includes_custom_event(self):
        """Interrupt should also emit CustomEvent for backward compatibility."""
        interrupt = MockInterrupt(value={"reason": "database_modification"})
        task = MockTask(interrupts=[interrupt])
        mock_state = MockAgentState(
            values={"messages": []},
            tasks=[task],
        )

        from ag_ui.core import RunAgentInput

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-123",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        result = await self.agent.prepare_stream(input_data, mock_state, mock_config)
        events = result["events_to_dispatch"]

        # Find CustomEvent with OnInterrupt name
        custom_event = None
        for event in events:
            if hasattr(event, "type") and event.type == EventType.CUSTOM:
                if hasattr(event, "name") and event.name == "on_interrupt":
                    custom_event = event
                    break

        self.assertIsNotNone(custom_event, "CustomEvent OnInterrupt not found")

    async def test_no_interrupt_continues_stream(self):
        """When no interrupts, prepare_stream should return a stream."""
        # Mock state without interrupts
        mock_state = MockAgentState(
            values={"messages": []},
            tasks=[],  # No tasks = no interrupts
        )
        self.mock_graph.astream_events = MagicMock(return_value=AsyncMock())

        from ag_ui.core import RunAgentInput

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-123",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        result = await self.agent.prepare_stream(input_data, mock_state, mock_config)

        # Should have stream, not events_to_dispatch
        self.assertIn("stream", result)
        self.assertNotIn("events_to_dispatch", result)


class TestResumeIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for resume handling."""

    def setUp(self):
        """Set up test fixtures."""
        from ag_ui_langgraph.agent import LangGraphAgent

        self.mock_graph = MagicMock()
        self.mock_graph.get_input_jsonschema.return_value = {"properties": {}}
        self.mock_graph.get_output_jsonschema.return_value = {"properties": {}}
        mock_config_schema = MagicMock()
        mock_config_schema.schema.return_value = {"properties": {}}
        self.mock_graph.config_schema.return_value = mock_config_schema
        self.mock_graph.aget_state = AsyncMock()
        self.mock_graph.astream_events = MagicMock(return_value=AsyncMock())

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)

        # Initialize active_run state
        self.agent.active_run = {
            "id": "run-123",
            "run_id": "run-123",
            "reasoning_process": None,
            "has_function_streaming": False,
            "mode": "start",
            "schema_keys": {"input": [], "output": [], "config": [], "context": []},
        }

    async def test_resume_with_interrupt_uses_command(self):
        """Resume with interrupt should use LangGraph Command."""
        # Mock state with interrupts (required for resume)
        interrupt = MockInterrupt(value={"reason": "human_approval"})
        task = MockTask(interrupts=[interrupt])
        mock_state = MockAgentState(
            values={"messages": []},
            tasks=[task],
        )

        from ag_ui.core import RunAgentInput, Resume

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-123",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
            resume=Resume(interrupt_id="int-123", payload={"approved": True}),
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        # Should not return events_to_dispatch when resume is provided
        result = await self.agent.prepare_stream(input_data, mock_state, mock_config)

        # With resume, should continue to stream (not return interrupt events)
        self.assertIn("stream", result)


if __name__ == "__main__":
    unittest.main()

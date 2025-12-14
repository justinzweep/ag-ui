"""
Tests for tool result handling during interrupt/resume flow.

These tests verify that AG-UI does NOT explicitly inject ToolMessages during resume,
as LangGraph's ToolNode automatically creates ToolMessages when the interrupted
tool function completes.
"""

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, call

from ag_ui.core import EventType, Resume, RunAgentInput


@dataclass
class MockInterrupt:
    """Mock LangGraph interrupt object with tool context."""

    id: str = "int-123"
    value: Any = None

    def __post_init__(self):
        if self.value is None:
            self.value = {
                "tool": "read_file",
                "tool_call_id": "tc-456",
                "reason": "human_approval",
            }


@dataclass
class MockTask:
    """Mock LangGraph task with interrupts."""

    name: str = "tools"  # Node name where the task is executing
    interrupts: List[MockInterrupt] = None

    def __post_init__(self):
        if self.interrupts is None:
            self.interrupts = []


@dataclass
class MockAgentState:
    """Mock LangGraph agent state."""

    values: Dict[str, Any] = None
    tasks: List[MockTask] = None
    metadata: Dict[str, Any] = None
    next: tuple = ()

    def __post_init__(self):
        if self.values is None:
            self.values = {"messages": [], "tools": []}
        if self.tasks is None:
            self.tasks = []
        if self.metadata is None:
            self.metadata = {"writes": {}}


class TestNoExplicitToolMessageInjection(unittest.TestCase):
    """Tests verifying AG-UI does NOT inject ToolMessages explicitly.

    LangGraph's ToolNode automatically creates ToolMessages when a tool function
    returns after an interrupt() resumes. Explicit injection would create duplicates.
    """

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
        self.mock_graph.aupdate_state = AsyncMock()
        self.mock_graph.astream_events = MagicMock(return_value=AsyncMock())

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)

    def test_resume_does_not_call_aupdate_state_for_tool_message(self):
        """Resume should NOT call aupdate_state to inject ToolMessage."""
        import asyncio

        async def run_test():
            # Set up agent state with an active interrupt containing tool context
            interrupt = MockInterrupt(
                id="int-abc",
                value={
                    "tool": "read_file",
                    "tool_call_id": "tc-123",
                    "reason": "human_approval",
                },
            )
            task = MockTask(interrupts=[interrupt])
            agent_state = MockAgentState(tasks=[task], values={"messages": []})
            config = {"configurable": {"thread_id": "thread-1"}}

            # Initialize active_run as _handle_stream_events does
            self.agent.active_run = {
                "id": "run-1",
                "thread_id": "thread-1",
                "reasoning_process": None,
                "node_name": "tool_node",
                "has_function_streaming": False,
                "mode": "start",
            }

            # Create resume with tool result
            resume = Resume(
                interrupt_id="int-abc",
                payload={"content": "file contents here", "success": True},
            )
            input_data = RunAgentInput(
                thread_id="thread-1",
                run_id="run-1",
                state={},
                messages=[],
                tools=[],
                context=[],
                forwarded_props={},
                resume=resume,
            )

            # Mock get_stream_kwargs to avoid actual streaming setup
            self.agent.get_stream_kwargs = MagicMock(return_value={})

            await self.agent.prepare_stream(input_data, agent_state, config)

            # Verify aupdate_state was NOT called with ToolMessage
            # LangGraph's ToolNode will handle creating the ToolMessage naturally
            for call_args in self.mock_graph.aupdate_state.call_args_list:
                if len(call_args[0]) > 1:
                    state_update = call_args[0][1]
                    if "messages" in state_update:
                        from langchain_core.messages import ToolMessage
                        for msg in state_update.get("messages", []):
                            self.assertNotIsInstance(
                                msg,
                                ToolMessage,
                                "ToolMessage should not be explicitly injected - "
                                "LangGraph ToolNode handles this automatically",
                            )

        asyncio.run(run_test())

    def test_resume_creates_command_with_resume_payload(self):
        """Resume should create a Command with the resume payload for astream_events."""
        import asyncio

        async def run_test():
            interrupt = MockInterrupt(
                id="int-abc",
                value={
                    "tool": "read_file",
                    "tool_call_id": "tc-123",
                },
            )
            task = MockTask(interrupts=[interrupt])
            agent_state = MockAgentState(tasks=[task], values={"messages": []})
            config = {"configurable": {"thread_id": "thread-1"}}

            self.agent.active_run = {
                "id": "run-1",
                "thread_id": "thread-1",
                "reasoning_process": None,
                "node_name": "tool_node",
                "has_function_streaming": False,
                "mode": "start",
            }

            resume = Resume(
                interrupt_id="int-abc",
                payload={"content": "file contents here"},
            )
            input_data = RunAgentInput(
                thread_id="thread-1",
                run_id="run-1",
                state={},
                messages=[],
                tools=[],
                context=[],
                forwarded_props={},
                resume=resume,
            )

            # Mock get_stream_kwargs to capture what input is passed
            captured_input = None

            def capture_stream_kwargs(input, config, subgraphs, version):
                nonlocal captured_input
                captured_input = input
                return {}

            self.agent.get_stream_kwargs = MagicMock(side_effect=capture_stream_kwargs)

            await self.agent.prepare_stream(input_data, agent_state, config)

            # Verify the stream input is a Command with the resume mapping
            from langgraph.types import Command
            self.assertIsInstance(captured_input, Command)
            # Command.resume should contain the interrupt_id mapping to payload
            self.assertIn("int-abc", captured_input.resume)
            self.assertEqual(
                captured_input.resume["int-abc"],
                {"content": "file contents here"},
            )

        asyncio.run(run_test())


class TestMultipleInterruptsErrorMessage(unittest.TestCase):
    """Tests for Issue 3: Improved error message for multiple interrupts."""

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
        self.mock_graph.aupdate_state = AsyncMock()

        self.agent = LangGraphAgent(name="test-agent", graph=self.mock_graph)

    def test_error_includes_interrupt_ids(self):
        """Error message should include the IDs of all pending interrupts."""
        import asyncio

        async def run_test():
            # Set up agent state with multiple active interrupts
            interrupt1 = MockInterrupt(id="int-aaa", value={"tool": "tool_a"})
            interrupt2 = MockInterrupt(id="int-bbb", value={"tool": "tool_b"})
            task = MockTask(interrupts=[interrupt1, interrupt2])
            agent_state = MockAgentState(tasks=[task], values={"messages": []})
            config = {"configurable": {"thread_id": "thread-1"}}

            self.agent.active_run = {
                "id": "run-1",
                "thread_id": "thread-1",
                "reasoning_process": None,
                "node_name": "tool_node",
                "has_function_streaming": False,
                "mode": "start",
            }

            # Resume WITHOUT interrupt_id when multiple interrupts pending
            resume = Resume(
                payload={"approved": True},
                # No interrupt_id specified
            )
            input_data = RunAgentInput(
                thread_id="thread-1",
                run_id="run-1",
                state={},
                messages=[],
                tools=[],
                context=[],
                forwarded_props={},
                resume=resume,
            )

            with self.assertRaises(ValueError) as context:
                await self.agent.prepare_stream(input_data, agent_state, config)

            error_msg = str(context.exception)
            # Should include count
            self.assertIn("2", error_msg)
            # Should include interrupt IDs
            self.assertIn("int-aaa", error_msg)
            self.assertIn("int-bbb", error_msg)
            # Should mention interruptId
            self.assertIn("interruptId", error_msg)

        asyncio.run(run_test())


class TestToolMessageWrapping(unittest.TestCase):
    """Tests for tool result content wrapping."""

    def test_wrap_tool_result_preserves_structure(self):
        """wrap_tool_result_content should properly format tool results."""
        from ag_ui_langgraph.utils import wrap_tool_result_content

        result = wrap_tool_result_content(
            tool_call_id="tc-123",
            tool_name="read_file",
            raw_content={"content": "file data", "path": "/test.txt"},
        )

        # Result should be a string (JSON)
        self.assertIsInstance(result, str)
        # Should contain the original content
        self.assertIn("file data", result)
        self.assertIn("/test.txt", result)


if __name__ == "__main__":
    unittest.main()

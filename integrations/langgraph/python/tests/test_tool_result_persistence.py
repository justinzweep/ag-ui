"""
Tests for tool result persistence during interrupt/resume flow.

These tests verify that when a client resumes an interrupted run with a tool result,
the result is properly persisted to LangGraph state as a ToolMessage.
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


class TestToolResultInjection(unittest.TestCase):
    """Tests for Issue 1: Tool results persisted to LangGraph state."""

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

    def test_resume_injects_tool_message(self):
        """Tool result should be injected as ToolMessage before resume."""
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

            # Verify aupdate_state was called to inject the ToolMessage
            self.mock_graph.aupdate_state.assert_called()
            call_args = self.mock_graph.aupdate_state.call_args

            # Check the state update contains a messages key with ToolMessage
            state_update = call_args[0][1]  # Second positional arg
            self.assertIn("messages", state_update)
            self.assertEqual(len(state_update["messages"]), 1)

            tool_message = state_update["messages"][0]
            self.assertEqual(tool_message.tool_call_id, "tc-123")
            self.assertEqual(tool_message.name, "read_file")

        asyncio.run(run_test())

    def test_resume_uses_task_name_for_as_node(self):
        """aupdate_state should use task name as as_node, not active_run['node_name']."""
        import asyncio

        async def run_test():
            # Set up agent state with task named "my_tools_node"
            interrupt = MockInterrupt(
                id="int-abc",
                value={
                    "tool": "read_file",
                    "tool_call_id": "tc-123",
                    "reason": "human_approval",
                },
            )
            task = MockTask(name="my_tools_node", interrupts=[interrupt])
            agent_state = MockAgentState(tasks=[task], values={"messages": []})
            config = {"configurable": {"thread_id": "thread-1"}}

            # Initialize active_run with a DIFFERENT node_name to verify it's not used
            self.agent.active_run = {
                "id": "run-1",
                "thread_id": "thread-1",
                "reasoning_process": None,
                "node_name": "wrong_node",  # This should NOT be used
                "has_function_streaming": False,
                "mode": "start",
            }

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

            self.agent.get_stream_kwargs = MagicMock(return_value={})

            await self.agent.prepare_stream(input_data, agent_state, config)

            # Verify aupdate_state was called with as_node="my_tools_node" (task name)
            self.mock_graph.aupdate_state.assert_called()
            call_args = self.mock_graph.aupdate_state.call_args

            # Check that as_node is the task name, not active_run["node_name"]
            as_node = call_args[1].get("as_node") if call_args[1] else call_args[0][2]
            self.assertEqual(
                as_node,
                "my_tools_node",
                "as_node should be the task name from agent_state.tasks[0].name",
            )

        asyncio.run(run_test())

    def test_resume_without_tool_context_skips_injection(self):
        """Resume without tool_call_id in interrupt should skip ToolMessage injection."""
        import asyncio

        async def run_test():
            # Set up agent state with an interrupt WITHOUT tool context
            interrupt = MockInterrupt(
                id="int-abc",
                value={
                    "reason": "human_approval",
                    # No tool or tool_call_id
                },
            )
            task = MockTask(interrupts=[interrupt])
            agent_state = MockAgentState(tasks=[task], values={"messages": []})
            config = {"configurable": {"thread_id": "thread-1"}}

            self.agent.active_run = {
                "id": "run-1",
                "thread_id": "thread-1",
                "reasoning_process": None,
                "node_name": "some_node",
                "has_function_streaming": False,
                "mode": "start",
            }

            resume = Resume(
                interrupt_id="int-abc",
                payload={"approved": True},
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

            self.agent.get_stream_kwargs = MagicMock(return_value={})

            await self.agent.prepare_stream(input_data, agent_state, config)

            # Verify aupdate_state was NOT called for ToolMessage injection
            # (it might be called for other reasons, but not with messages containing ToolMessage)
            for call_args in self.mock_graph.aupdate_state.call_args_list:
                state_update = call_args[0][1] if len(call_args[0]) > 1 else {}
                if "messages" in state_update:
                    # If messages is in the update, it shouldn't be from tool injection
                    # since there's no tool_call_id in the interrupt
                    from langchain_core.messages import ToolMessage

                    for msg in state_update.get("messages", []):
                        self.assertNotIsInstance(
                            msg,
                            ToolMessage,
                            "ToolMessage should not be injected without tool_call_id",
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

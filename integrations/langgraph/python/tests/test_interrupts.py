"""
Tests for interrupt detection and resume handling.
"""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ag_ui.core import EventType, RunAgentInput, Resume


@dataclass
class MockInterrupt:
    """Mock LangGraph interrupt object."""

    value: Any = None


@dataclass
class MockTask:
    """Mock LangGraph task with interrupts."""

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


class TestGetInterruptReason(unittest.TestCase):
    """Tests for _get_interrupt_reason() method."""

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

    def test_extract_reason_from_dict_value(self):
        """Should extract reason from interrupt.value dict."""
        interrupt = MockInterrupt(value={"reason": "human_approval", "payload": {}})

        result = self.agent._get_interrupt_reason([interrupt])

        self.assertEqual(result, "human_approval")

    def test_extract_custom_reason(self):
        """Should extract custom reason string."""
        interrupt = MockInterrupt(value={"reason": "database_modification"})

        result = self.agent._get_interrupt_reason([interrupt])

        self.assertEqual(result, "database_modification")

    def test_default_reason_when_missing(self):
        """Should return human_approval when reason not in dict."""
        interrupt = MockInterrupt(value={"payload": "some data"})

        result = self.agent._get_interrupt_reason([interrupt])

        self.assertEqual(result, "human_approval")

    def test_default_reason_for_string_value(self):
        """Should return human_approval for non-dict interrupt value."""
        interrupt = MockInterrupt(value="string interrupt value")

        result = self.agent._get_interrupt_reason([interrupt])

        self.assertEqual(result, "human_approval")

    def test_none_for_empty_interrupts(self):
        """Should return None for empty interrupt list."""
        result = self.agent._get_interrupt_reason([])

        self.assertIsNone(result)

    def test_none_for_none_interrupts(self):
        """Should return None for None interrupts."""
        result = self.agent._get_interrupt_reason(None)

        self.assertIsNone(result)

    def test_uses_first_interrupt(self):
        """Should use the first interrupt when multiple exist."""
        interrupt1 = MockInterrupt(value={"reason": "first_reason"})
        interrupt2 = MockInterrupt(value={"reason": "second_reason"})

        result = self.agent._get_interrupt_reason([interrupt1, interrupt2])

        self.assertEqual(result, "first_reason")


class TestInterruptDetection(unittest.TestCase):
    """Tests for interrupt detection from agent state."""

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

    def test_detect_interrupt_from_tasks(self):
        """Should detect interrupts from task state."""
        interrupt = MockInterrupt(value={"reason": "human_approval"})
        task = MockTask(interrupts=[interrupt])
        state = MockAgentState(tasks=[task])

        interrupts = state.tasks[0].interrupts if state.tasks else []

        self.assertEqual(len(interrupts), 1)
        self.assertEqual(interrupts[0].value["reason"], "human_approval")

    def test_no_interrupt_when_tasks_empty(self):
        """Should not detect interrupt when tasks are empty."""
        state = MockAgentState(tasks=[])

        tasks = state.tasks if len(state.tasks) > 0 else None
        interrupts = tasks[0].interrupts if tasks else []

        self.assertEqual(interrupts, [])

    def test_no_interrupt_when_task_has_no_interrupts(self):
        """Should not detect interrupt when task has empty interrupts."""
        task = MockTask(interrupts=[])
        state = MockAgentState(tasks=[task])

        interrupts = state.tasks[0].interrupts if state.tasks else []

        self.assertEqual(len(interrupts), 0)


class TestResumeHandling(unittest.TestCase):
    """Tests for resume input handling."""

    def test_resume_from_input_resume_field(self):
        """Should extract resume payload from input.resume field."""
        resume = Resume(interrupt_id="int-123", payload={"approved": True})
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

        resume_input = None
        if input_data.resume and input_data.resume.payload is not None:
            resume_input = input_data.resume.payload

        self.assertEqual(resume_input, {"approved": True})

    def test_resume_from_forwarded_props(self):
        """Should extract resume payload from forwardedProps.command.resume."""
        forwarded_props = {"command": {"resume": {"approved": True}}}

        resume_input = forwarded_props.get("command", {}).get("resume", None)

        self.assertEqual(resume_input, {"approved": True})

    def test_input_resume_takes_precedence(self):
        """input.resume should take precedence over forwardedProps."""
        resume = Resume(interrupt_id="int-123", payload={"from": "input.resume"})
        input_data = RunAgentInput(
            thread_id="thread-1",
            run_id="run-1",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={"command": {"resume": {"from": "forwarded_props"}}},
            resume=resume,
        )

        resume_input = None
        if input_data.resume and input_data.resume.payload is not None:
            resume_input = input_data.resume.payload
        elif input_data.forwarded_props:
            resume_input = input_data.forwarded_props.get("command", {}).get("resume", None)

        self.assertEqual(resume_input, {"from": "input.resume"})

    def test_no_resume_returns_none(self):
        """Should return None when no resume payload exists."""
        input_data = RunAgentInput(
            thread_id="thread-1",
            run_id="run-1",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )

        resume_input = None
        if input_data.resume and input_data.resume.payload is not None:
            resume_input = input_data.resume.payload
        elif input_data.forwarded_props:
            resume_input = input_data.forwarded_props.get("command", {}).get("resume", None)

        self.assertIsNone(resume_input)


class TestRunFinishedEventWithInterrupt(unittest.TestCase):
    """Tests for RunFinishedEvent with interrupt outcome."""

    def test_run_finished_interrupt_schema(self):
        """Should create valid RunFinishedEvent with interrupt outcome."""
        from ag_ui.core import RunFinishedEvent, Interrupt

        event = RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id="thread-1",
            run_id="run-1",
            outcome="interrupt",
            interrupt=Interrupt(id="int-123", reason="human_approval", payload={"tool": "send_email"}),
        )

        self.assertEqual(event.type, EventType.RUN_FINISHED)
        self.assertEqual(event.outcome, "interrupt")
        self.assertEqual(event.interrupt.id, "int-123")
        self.assertEqual(event.interrupt.reason, "human_approval")

    def test_run_finished_success_schema(self):
        """Should create valid RunFinishedEvent with success outcome."""
        from ag_ui.core import RunFinishedEvent

        event = RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id="thread-1",
            run_id="run-1",
            outcome="success",
            result={"message": "completed"},
        )

        self.assertEqual(event.type, EventType.RUN_FINISHED)
        self.assertEqual(event.outcome, "success")
        self.assertEqual(event.result, {"message": "completed"})

    def test_run_finished_backward_compatible(self):
        """Should allow RunFinishedEvent without outcome for backward compatibility."""
        from ag_ui.core import RunFinishedEvent

        event = RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id="thread-1",
            run_id="run-1",
            result={"data": "legacy"},
        )

        self.assertEqual(event.type, EventType.RUN_FINISHED)
        self.assertIsNone(event.outcome)


class TestPrepareStreamResumeHandling(unittest.TestCase):
    """Tests for prepare_stream method's resume handling."""

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

    async def _prepare_stream_with_active_interrupt(self, input_data):
        """Helper to call prepare_stream with an active interrupt state."""
        interrupt = MockInterrupt(value={"reason": "human_approval"})
        task = MockTask(interrupts=[interrupt])
        agent_state = MockAgentState(tasks=[task], values={"messages": []})
        config = {"configurable": {"thread_id": input_data.thread_id}}

        # Initialize active_run as _handle_stream_events does
        self.agent.active_run = {
            "id": input_data.run_id,
            "thread_id": input_data.thread_id,
            "reasoning_process": None,
            "node_name": None,
            "has_function_streaming": False,
            "mode": "start",
        }

        return await self.agent.prepare_stream(input_data, agent_state, config)

    def test_prepare_stream_recognizes_input_resume(self):
        """prepare_stream should recognize resume from input.resume field, not just forwarded_props."""
        import asyncio

        async def run_test():
            resume = Resume(interrupt_id="int-123", payload={"approved": True})
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

            result = await self._prepare_stream_with_active_interrupt(input_data)

            # When resume_input is recognized, prepare_stream should NOT return events_to_dispatch
            # because the request is a legitimate resume, not a "re-fetch pending interrupt" request
            events_to_dispatch = result.get("events_to_dispatch")

            # If events_to_dispatch is set, it means prepare_stream didn't recognize the resume
            # which is the bug we're fixing
            self.assertIsNone(events_to_dispatch,
                "prepare_stream should recognize input.resume and not return events_to_dispatch")

        asyncio.run(run_test())


class TestCompleteInterruptResumeFlow(unittest.TestCase):
    """Integration tests for complete interrupt/resume flow."""

    def test_interrupt_to_resume_flow(self):
        """Should validate complete interrupt to resume flow."""
        from ag_ui.core import RunFinishedEvent, Interrupt, RunAgentInput, Resume

        # Step 1: Agent sends interrupt
        interrupt_event = RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id="thread-1",
            run_id="run-1",
            outcome="interrupt",
            interrupt=Interrupt(
                id="int-abc123",
                reason="human_approval",
                payload={"tool": "send_email", "args": {"to": "test@example.com"}},
            ),
        )

        self.assertEqual(interrupt_event.outcome, "interrupt")
        self.assertIsNotNone(interrupt_event.interrupt)

        # Step 2: User responds with approval
        resume_input = RunAgentInput(
            thread_id="thread-1",
            run_id="run-2",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
            resume=Resume(interrupt_id="int-abc123", payload={"approved": True}),
        )

        self.assertEqual(resume_input.thread_id, interrupt_event.thread_id)
        self.assertEqual(resume_input.resume.interrupt_id, interrupt_event.interrupt.id)

        # Step 3: Agent completes successfully
        success_event = RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id="thread-1",
            run_id="run-2",
            outcome="success",
            result={"email_sent": True},
        )

        self.assertEqual(success_event.outcome, "success")
        self.assertEqual(success_event.thread_id, resume_input.thread_id)


if __name__ == "__main__":
    unittest.main()

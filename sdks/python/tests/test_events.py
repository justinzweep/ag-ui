import unittest
from datetime import datetime

from ag_ui.core.events import (
    ActivityDeltaEvent,
    ActivitySnapshotEvent,
    BaseEvent,
    CustomEvent,
    Event,
    EventType,
    MessagesSnapshotEvent,
    RawEvent,
    ReasoningEndEvent,
    ReasoningMessageChunkEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ag_ui.core.types import AssistantMessage, FunctionCall, ToolCall, UserMessage
from pydantic import TypeAdapter


class TestEvents(unittest.TestCase):
    """Test suite for event classes"""

    def test_event_types_enum(self):
        """Test the EventType enum values"""
        self.assertEqual(EventType.TEXT_MESSAGE_START.value, "TEXT_MESSAGE_START")
        self.assertEqual(EventType.TOOL_CALL_ARGS.value, "TOOL_CALL_ARGS")
        self.assertEqual(EventType.STATE_SNAPSHOT.value, "STATE_SNAPSHOT")
        self.assertEqual(EventType.RUN_ERROR.value, "RUN_ERROR")
        self.assertEqual(EventType.STEP_FINISHED.value, "STEP_FINISHED")

    def test_base_event_creation(self):
        """Test creating a BaseEvent instance"""
        timestamp = int(datetime.now().timestamp() * 1000)
        event = BaseEvent(type=EventType.RAW, timestamp=timestamp)
        self.assertEqual(event.type, EventType.RAW)
        self.assertEqual(event.timestamp, timestamp)
        self.assertIsNone(event.raw_event)

    def test_text_message_start(self):
        """Test creating and serializing a TextMessageStartEvent event"""
        event = TextMessageStartEvent(message_id="msg_123", timestamp=1648214400000)
        self.assertEqual(event.message_id, "msg_123")
        self.assertEqual(event.role, "assistant")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "TEXT_MESSAGE_START")
        self.assertEqual(serialized["messageId"], "msg_123")
        self.assertEqual(serialized["timestamp"], 1648214400000)

    def test_text_message_content(self):
        """Test creating and serializing a TextMessageContentEvent event"""
        event = TextMessageContentEvent(
            message_id="msg_123", delta="Hello, world!", timestamp=1648214400000
        )
        self.assertEqual(event.message_id, "msg_123")
        self.assertEqual(event.delta, "Hello, world!")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "TEXT_MESSAGE_CONTENT")
        self.assertEqual(serialized["messageId"], "msg_123")
        self.assertEqual(serialized["delta"], "Hello, world!")

    def test_text_message_end(self):
        """Test creating and serializing a TextMessageEndEvent event"""
        event = TextMessageEndEvent(message_id="msg_123", timestamp=1648214400000)
        self.assertEqual(event.message_id, "msg_123")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "TEXT_MESSAGE_END")
        self.assertEqual(serialized["messageId"], "msg_123")

    def test_reasoning_start(self):
        """Test creating and serializing a ReasoningStartEvent event"""
        event = ReasoningStartEvent(message_id="reasoning_123", timestamp=1648214400000)
        self.assertEqual(event.message_id, "reasoning_123")
        self.assertIsNone(event.encrypted_content)

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "REASONING_START")
        self.assertEqual(serialized["messageId"], "reasoning_123")

        # Test with encrypted content
        event_encrypted = ReasoningStartEvent(
            message_id="reasoning_456",
            encrypted_content="encrypted_blob_xyz",
            timestamp=1648214400000,
        )
        self.assertEqual(event_encrypted.encrypted_content, "encrypted_blob_xyz")
        serialized_encrypted = event_encrypted.model_dump(by_alias=True)
        self.assertEqual(serialized_encrypted["encryptedContent"], "encrypted_blob_xyz")

    def test_reasoning_message_start(self):
        """Test creating and serializing a ReasoningMessageStartEvent event"""
        event = ReasoningMessageStartEvent(
            message_id="msg_reasoning_123", timestamp=1648214400000
        )
        self.assertEqual(event.message_id, "msg_reasoning_123")
        self.assertEqual(event.role, "assistant")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "REASONING_MESSAGE_START")
        self.assertEqual(serialized["messageId"], "msg_reasoning_123")
        self.assertEqual(serialized["role"], "assistant")

    def test_reasoning_message_content(self):
        """Test creating and serializing a ReasoningMessageContentEvent event"""
        event = ReasoningMessageContentEvent(
            message_id="msg_reasoning_123",
            delta="Thinking step...",
            timestamp=1648214400000,
        )
        self.assertEqual(event.message_id, "msg_reasoning_123")
        self.assertEqual(event.delta, "Thinking step...")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "REASONING_MESSAGE_CONTENT")
        self.assertEqual(serialized["messageId"], "msg_reasoning_123")
        self.assertEqual(serialized["delta"], "Thinking step...")

    def test_reasoning_message_end(self):
        """Test creating and serializing a ReasoningMessageEndEvent event"""
        event = ReasoningMessageEndEvent(
            message_id="msg_reasoning_123", timestamp=1648214400000
        )
        self.assertEqual(event.message_id, "msg_reasoning_123")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "REASONING_MESSAGE_END")
        self.assertEqual(serialized["messageId"], "msg_reasoning_123")

    def test_reasoning_message_chunk(self):
        """Test creating and serializing a ReasoningMessageChunkEvent event"""
        # Test with both messageId and delta
        event = ReasoningMessageChunkEvent(
            message_id="msg_reasoning_456",
            delta="Chunk content",
            timestamp=1648214400000,
        )
        self.assertEqual(event.message_id, "msg_reasoning_456")
        self.assertEqual(event.delta, "Chunk content")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "REASONING_MESSAGE_CHUNK")
        self.assertEqual(serialized["messageId"], "msg_reasoning_456")
        self.assertEqual(serialized["delta"], "Chunk content")

        # Test with optional fields as None
        event_minimal = ReasoningMessageChunkEvent(timestamp=1648214400000)
        self.assertIsNone(event_minimal.message_id)
        self.assertIsNone(event_minimal.delta)

    def test_reasoning_end(self):
        """Test creating and serializing a ReasoningEndEvent event"""
        event = ReasoningEndEvent(message_id="reasoning_123", timestamp=1648214400000)
        self.assertEqual(event.message_id, "reasoning_123")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "REASONING_END")
        self.assertEqual(serialized["messageId"], "reasoning_123")

    def test_tool_call_start(self):
        """Test creating and serializing a ToolCallStartEvent event"""
        event = ToolCallStartEvent(
            tool_call_id="call_123",
            tool_call_name="get_weather",
            parent_message_id="msg_456",
            timestamp=1648214400000,
        )
        self.assertEqual(event.tool_call_id, "call_123")
        self.assertEqual(event.tool_call_name, "get_weather")
        self.assertEqual(event.parent_message_id, "msg_456")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "TOOL_CALL_START")
        self.assertEqual(serialized["toolCallId"], "call_123")
        self.assertEqual(serialized["toolCallName"], "get_weather")
        self.assertEqual(serialized["parentMessageId"], "msg_456")

    def test_tool_call_args(self):
        """Test creating and serializing a ToolCallArgsEvent event"""
        event = ToolCallArgsEvent(
            tool_call_id="call_123",
            delta='{"location": "New York"}',
            timestamp=1648214400000,
        )
        self.assertEqual(event.tool_call_id, "call_123")
        self.assertEqual(event.delta, '{"location": "New York"}')

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "TOOL_CALL_ARGS")
        self.assertEqual(serialized["toolCallId"], "call_123")
        self.assertEqual(serialized["delta"], '{"location": "New York"}')

    def test_tool_call_end(self):
        """Test creating and serializing a ToolCallEndEvent event"""
        event = ToolCallEndEvent(tool_call_id="call_123", timestamp=1648214400000)
        self.assertEqual(event.tool_call_id, "call_123")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "TOOL_CALL_END")
        self.assertEqual(serialized["toolCallId"], "call_123")

    def test_state_snapshot(self):
        """Test creating and serializing a StateSnapshotEvent event"""
        state = {"conversation_state": "active", "user_info": {"name": "John"}}
        event = StateSnapshotEvent(snapshot=state, timestamp=1648214400000)
        self.assertEqual(event.snapshot, state)

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "STATE_SNAPSHOT")
        self.assertEqual(serialized["snapshot"]["conversation_state"], "active")
        self.assertEqual(serialized["snapshot"]["user_info"]["name"], "John")

    def test_state_delta(self):
        """Test creating and serializing a StateDeltaEvent event"""
        # JSON Patch format
        delta = [
            {"op": "replace", "path": "/conversation_state", "value": "paused"},
            {"op": "add", "path": "/user_info/age", "value": 30},
        ]
        event = StateDeltaEvent(delta=delta, timestamp=1648214400000)
        self.assertEqual(event.delta, delta)

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "STATE_DELTA")
        self.assertEqual(len(serialized["delta"]), 2)
        self.assertEqual(serialized["delta"][0]["op"], "replace")
        self.assertEqual(serialized["delta"][1]["path"], "/user_info/age")

    def test_messages_snapshot(self):
        """Test creating and serializing a MessagesSnapshotEvent event"""
        messages = [
            UserMessage(id="user_1", content="Hello"),
            AssistantMessage(
                id="asst_1",
                content="Hi there",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="get_weather", arguments='{"location": "New York"}'
                        ),
                    )
                ],
            ),
        ]
        event = MessagesSnapshotEvent(messages=messages, timestamp=1648214400000)
        self.assertEqual(len(event.messages), 2)
        self.assertEqual(event.messages[0].id, "user_1")
        self.assertEqual(event.messages[1].tool_calls[0].function.name, "get_weather")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "MESSAGES_SNAPSHOT")
        self.assertEqual(len(serialized["messages"]), 2)
        self.assertEqual(serialized["messages"][0]["role"], "user")
        self.assertEqual(
            serialized["messages"][1]["toolCalls"][0]["function"]["name"], "get_weather"
        )

    def test_activity_snapshot(self):
        """Test creating and serializing an ActivitySnapshotEvent"""
        content = {"tasks": ["search", "summarize"]}
        event = ActivitySnapshotEvent(
            message_id="msg_activity",
            activity_type="PLAN",
            content=content,
            timestamp=1648214400000,
        )

        self.assertEqual(event.message_id, "msg_activity")
        self.assertEqual(event.activity_type, "PLAN")
        self.assertEqual(event.content, content)
        self.assertTrue(event.replace)

        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "ACTIVITY_SNAPSHOT")
        self.assertEqual(serialized["messageId"], "msg_activity")
        self.assertEqual(serialized["activityType"], "PLAN")
        self.assertEqual(serialized["content"], content)
        self.assertTrue(serialized["replace"])

        event_replace_false = ActivitySnapshotEvent(
            message_id="msg_activity",
            activity_type="PLAN",
            content=content,
            replace=False,
        )
        self.assertFalse(event_replace_false.replace)
        serialized_false = event_replace_false.model_dump(by_alias=True)
        self.assertFalse(serialized_false["replace"])

    def test_activity_delta(self):
        """Test creating and serializing an ActivityDeltaEvent"""
        patch = [{"op": "replace", "path": "/tasks/0", "value": "✓ search"}]
        event = ActivityDeltaEvent(
            message_id="msg_activity",
            activity_type="PLAN",
            patch=patch,
            timestamp=1648214400000,
        )

        self.assertEqual(event.message_id, "msg_activity")
        self.assertEqual(event.activity_type, "PLAN")
        self.assertEqual(event.patch, patch)

        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "ACTIVITY_DELTA")
        self.assertEqual(serialized["messageId"], "msg_activity")
        self.assertEqual(serialized["activityType"], "PLAN")
        self.assertEqual(serialized["patch"], patch)

    def test_raw_event(self):
        """Test creating and serializing a RawEvent"""
        raw_data = {"origin": "server", "data": {"key": "value"}}
        event = RawEvent(event=raw_data, source="api", timestamp=1648214400000)
        self.assertEqual(event.event, raw_data)
        self.assertEqual(event.source, "api")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "RAW")
        self.assertEqual(serialized["event"]["origin"], "server")
        self.assertEqual(serialized["source"], "api")

    def test_custom_event(self):
        """Test creating and serializing a CustomEvent"""
        event = CustomEvent(
            name="user_action",
            value={"action": "click", "element": "button"},
            timestamp=1648214400000,
        )
        self.assertEqual(event.name, "user_action")
        self.assertEqual(event.value["action"], "click")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "CUSTOM")
        self.assertEqual(serialized["name"], "user_action")
        self.assertEqual(serialized["value"]["element"], "button")

    def test_run_started(self):
        """Test creating and serializing a RunStartedEvent event"""
        event = RunStartedEvent(
            thread_id="thread_123", run_id="run_456", timestamp=1648214400000
        )
        self.assertEqual(event.thread_id, "thread_123")
        self.assertEqual(event.run_id, "run_456")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "RUN_STARTED")
        self.assertEqual(serialized["threadId"], "thread_123")
        self.assertEqual(serialized["runId"], "run_456")

    def test_run_finished(self):
        """Test creating and serializing a RunFinishedEvent event"""
        event = RunFinishedEvent(
            thread_id="thread_123", run_id="run_456", timestamp=1648214400000
        )
        self.assertEqual(event.thread_id, "thread_123")
        self.assertEqual(event.run_id, "run_456")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "RUN_FINISHED")
        self.assertEqual(serialized["threadId"], "thread_123")
        self.assertEqual(serialized["runId"], "run_456")

    def test_run_error(self):
        """Test creating and serializing a RunErrorEvent event"""
        event = RunErrorEvent(
            message="An error occurred during execution",
            code="ERROR_001",
            timestamp=1648214400000,
        )
        self.assertEqual(event.message, "An error occurred during execution")
        self.assertEqual(event.code, "ERROR_001")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "RUN_ERROR")
        self.assertEqual(serialized["message"], "An error occurred during execution")
        self.assertEqual(serialized["code"], "ERROR_001")

    def test_step_started(self):
        """Test creating and serializing a StepStartedEvent event"""
        event = StepStartedEvent(step_name="process_data", timestamp=1648214400000)
        self.assertEqual(event.step_name, "process_data")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "STEP_STARTED")
        self.assertEqual(serialized["stepName"], "process_data")

    def test_step_finished(self):
        """Test creating and serializing a StepFinishedEvent event"""
        event = StepFinishedEvent(step_name="process_data", timestamp=1648214400000)
        self.assertEqual(event.step_name, "process_data")

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "STEP_FINISHED")
        self.assertEqual(serialized["stepName"], "process_data")

    def test_event_union_deserialization(self):
        """Test the Event union type correctly deserializes different event types"""
        event_adapter = TypeAdapter(Event)

        # Test different event types
        event_data = [
            {
                "type": "TEXT_MESSAGE_START",
                "messageId": "msg_start",
                "role": "assistant",
                "timestamp": 1648214400000,
            },
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": "msg_content",
                "delta": "Hello!",
                "timestamp": 1648214400000,
            },
            {
                "type": "REASONING_START",
                "messageId": "reasoning_start",
                "timestamp": 1648214400000,
            },
            {
                "type": "REASONING_MESSAGE_START",
                "messageId": "msg_reasoning_start",
                "role": "assistant",
                "timestamp": 1648214400000,
            },
            {
                "type": "REASONING_MESSAGE_CONTENT",
                "messageId": "msg_reasoning_content",
                "delta": "Thinking...",
                "timestamp": 1648214400000,
            },
            {
                "type": "REASONING_MESSAGE_END",
                "messageId": "msg_reasoning_end",
                "timestamp": 1648214400000,
            },
            {
                "type": "REASONING_END",
                "messageId": "reasoning_end",
                "timestamp": 1648214400000,
            },
            {
                "type": "TOOL_CALL_START",
                "toolCallId": "call_start",
                "toolCallName": "get_info",
                "timestamp": 1648214400000,
            },
            {
                "type": "STATE_SNAPSHOT",
                "snapshot": {"status": "active"},
                "timestamp": 1648214400000,
            },
            {
                "type": "ACTIVITY_SNAPSHOT",
                "messageId": "msg_activity",
                "activityType": "PLAN",
                "content": {"tasks": []},
                "timestamp": 1648214400000,
            },
            {
                "type": "RUN_ERROR",
                "message": "Error occurred",
                "code": "ERR_001",
                "timestamp": 1648214400000,
            },
        ]

        expected_types = [
            TextMessageStartEvent,
            TextMessageContentEvent,
            ReasoningStartEvent,
            ReasoningMessageStartEvent,
            ReasoningMessageContentEvent,
            ReasoningMessageEndEvent,
            ReasoningEndEvent,
            ToolCallStartEvent,
            StateSnapshotEvent,
            ActivitySnapshotEvent,
            RunErrorEvent,
        ]

        for data, expected_type in zip(event_data, expected_types):
            event = event_adapter.validate_python(data)
            self.assertIsInstance(event, expected_type)
            self.assertEqual(event.type.value, data["type"])
            self.assertEqual(event.timestamp, data["timestamp"])

    def test_validation_constraints(self):
        """Test validation constraints for different event types"""
        # TextMessageContentEvent delta cannot be empty
        with self.assertRaises(ValueError):
            TextMessageContentEvent(
                message_id="msg_123",
                delta="",  # Empty delta, should fail
            )

        # ReasoningMessageContentEvent delta cannot be empty
        with self.assertRaises(ValueError):
            ReasoningMessageContentEvent(
                message_id="msg_reasoning_123",
                delta="",  # Empty delta, should fail
            )

    def test_serialization_round_trip(self):
        """Test serialization and deserialization for different event types"""
        # Create events of different types
        events = [
            TextMessageStartEvent(
                message_id="msg_123",
            ),
            TextMessageContentEvent(message_id="msg_123", delta="Hello, world!"),
            ReasoningStartEvent(
                message_id="reasoning_123", encrypted_content="encrypted_xyz"
            ),
            ReasoningMessageStartEvent(message_id="msg_reasoning_123"),
            ReasoningMessageContentEvent(
                message_id="msg_reasoning_123", delta="Thinking..."
            ),
            ReasoningMessageEndEvent(message_id="msg_reasoning_123"),
            ReasoningEndEvent(message_id="reasoning_123"),
            ToolCallStartEvent(tool_call_id="call_123", tool_call_name="get_weather"),
            StateSnapshotEvent(snapshot={"status": "active"}),
            MessagesSnapshotEvent(messages=[UserMessage(id="user_1", content="Hello")]),
            ActivitySnapshotEvent(
                message_id="msg_activity",
                activity_type="PLAN",
                content={"tasks": []},
            ),
            ActivityDeltaEvent(
                message_id="msg_activity",
                activity_type="PLAN",
                patch=[{"op": "add", "path": "/tasks/-", "value": "search"}],
            ),
            RunStartedEvent(thread_id="thread_123", run_id="run_456"),
        ]

        event_adapter = TypeAdapter(Event)

        # Test round trip for each event
        for original_event in events:
            # Serialize to JSON
            json_str = original_event.model_dump_json(by_alias=True)

            # Deserialize back to object
            deserialized_event = event_adapter.validate_json(json_str)

            # Verify the types match
            self.assertIsInstance(deserialized_event, type(original_event))
            self.assertEqual(deserialized_event.type, original_event.type)

            # Verify event-specific fields
            if isinstance(original_event, TextMessageStartEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
                self.assertEqual(deserialized_event.role, original_event.role)
            elif isinstance(original_event, TextMessageContentEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
                self.assertEqual(deserialized_event.delta, original_event.delta)
            elif isinstance(original_event, ReasoningStartEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
                self.assertEqual(
                    deserialized_event.encrypted_content,
                    original_event.encrypted_content,
                )
            elif isinstance(original_event, ReasoningMessageStartEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
                self.assertEqual(deserialized_event.role, original_event.role)
            elif isinstance(original_event, ReasoningMessageContentEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
                self.assertEqual(deserialized_event.delta, original_event.delta)
            elif isinstance(original_event, ReasoningMessageEndEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
            elif isinstance(original_event, ReasoningEndEvent):
                self.assertEqual(
                    deserialized_event.message_id, original_event.message_id
                )
            elif isinstance(original_event, ToolCallStartEvent):
                self.assertEqual(
                    deserialized_event.tool_call_id, original_event.tool_call_id
                )
                self.assertEqual(
                    deserialized_event.tool_call_name, original_event.tool_call_name
                )
            elif isinstance(original_event, StateSnapshotEvent):
                self.assertEqual(deserialized_event.snapshot, original_event.snapshot)
            elif isinstance(original_event, MessagesSnapshotEvent):
                self.assertEqual(
                    len(deserialized_event.messages), len(original_event.messages)
                )
                self.assertEqual(
                    deserialized_event.messages[0].id, original_event.messages[0].id
                )
            elif isinstance(original_event, RunStartedEvent):
                self.assertEqual(deserialized_event.thread_id, original_event.thread_id)
                self.assertEqual(deserialized_event.run_id, original_event.run_id)

    def test_raw_event_with_null_source(self):
        """Test RawEvent with null source"""
        event = RawEvent(
            event={"data": "test"},
            source=None,  # Explicit None
        )
        self.assertIsNone(event.source)

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["type"], "RAW")
        self.assertEqual(serialized["event"]["data"], "test")
        self.assertIsNone(serialized["source"])

        # Test round-trip
        event_adapter = TypeAdapter(Event)
        json_str = event.model_dump_json(by_alias=True)
        deserialized = event_adapter.validate_json(json_str)
        self.assertIsNone(deserialized.source)

    def test_complex_nested_event_structures(self):
        """Test complex nested structures within events"""
        # Complex state with nested objects and arrays
        complex_state = {
            "session": {
                "user": {
                    "id": "user_123",
                    "preferences": {
                        "theme": "dark",
                        "notifications": True,
                        "filters": ["news", "social", "tech"],
                    },
                },
                "stats": {
                    "messages": 42,
                    "interactions": {"clicks": 18, "searches": 7},
                },
            },
            "active_tools": ["search", "calculator", "weather"],
            "settings": {"language": "en", "timezone": "UTC-5"},
        }

        event = StateSnapshotEvent(snapshot=complex_state, timestamp=1648214400000)

        # Verify complex state structure
        self.assertEqual(event.snapshot["session"]["user"]["id"], "user_123")
        self.assertEqual(
            event.snapshot["session"]["user"]["preferences"]["theme"], "dark"
        )
        self.assertEqual(
            event.snapshot["session"]["stats"]["interactions"]["searches"], 7
        )
        self.assertEqual(event.snapshot["active_tools"][1], "calculator")

        # Test serialization and deserialization
        event_adapter = TypeAdapter(Event)
        json_str = event.model_dump_json(by_alias=True)
        deserialized = event_adapter.validate_json(json_str)

        # Verify structure is preserved
        self.assertEqual(
            deserialized.snapshot["session"]["user"]["preferences"]["filters"],
            ["news", "social", "tech"],
        )
        self.assertEqual(deserialized.snapshot["settings"]["timezone"], "UTC-5")

    def test_event_with_unicode_and_special_chars(self):
        """Test events with Unicode and special characters"""
        # Text with Unicode and special characters
        text = "Hello 你好 こんにちは 안녕하세요 👋 🌍 \n\t\"'\\/<>{}[]"

        event = TextMessageContentEvent(
            message_id="msg_unicode", delta=text, timestamp=1648214400000
        )

        # Verify text is stored correctly
        self.assertEqual(event.delta, text)

        # Test serialization and deserialization
        event_adapter = TypeAdapter(Event)
        json_str = event.model_dump_json(by_alias=True)
        deserialized = event_adapter.validate_json(json_str)

        # Verify Unicode and special characters are preserved
        self.assertEqual(deserialized.delta, text)


class TestInterruptLifecycle(unittest.TestCase):
    """Test suite for interrupt-aware run lifecycle events"""

    def test_run_finished_with_success_outcome(self):
        """Test RunFinishedEvent with outcome: success and result"""
        event = RunFinishedEvent(
            thread_id="thread_1",
            run_id="run_1",
            outcome="success",
            result={"data": "completed"},
        )
        self.assertEqual(event.type, EventType.RUN_FINISHED)
        self.assertEqual(event.outcome, "success")
        self.assertEqual(event.result, {"data": "completed"})
        self.assertIsNone(event.interrupt)

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["outcome"], "success")
        self.assertEqual(serialized["result"], {"data": "completed"})

    def test_run_finished_with_interrupt_outcome(self):
        """Test RunFinishedEvent with outcome: interrupt and interrupt object"""
        from ag_ui.core.events import Interrupt

        interrupt = Interrupt(
            id="int-abc123",
            reason="human_approval",
            payload={
                "proposal": {
                    "tool": "sendEmail",
                    "args": {"to": "a@b.com", "subject": "Hi"},
                }
            },
        )

        event = RunFinishedEvent(
            thread_id="thread_1",
            run_id="run_1",
            outcome="interrupt",
            interrupt=interrupt,
        )
        self.assertEqual(event.type, EventType.RUN_FINISHED)
        self.assertEqual(event.outcome, "interrupt")
        self.assertEqual(event.interrupt.id, "int-abc123")
        self.assertEqual(event.interrupt.reason, "human_approval")
        self.assertIsNone(event.result)

        # Test serialization
        serialized = event.model_dump(by_alias=True)
        self.assertEqual(serialized["outcome"], "interrupt")
        self.assertEqual(serialized["interrupt"]["id"], "int-abc123")
        self.assertEqual(serialized["interrupt"]["reason"], "human_approval")

    def test_run_finished_without_outcome_backward_compat(self):
        """Test RunFinishedEvent without outcome (backward compatibility)"""
        event = RunFinishedEvent(
            thread_id="thread_1",
            run_id="run_1",
            result={"data": "completed"},
        )
        self.assertEqual(event.type, EventType.RUN_FINISHED)
        self.assertIsNone(event.outcome)
        self.assertIsNone(event.interrupt)
        self.assertEqual(event.result, {"data": "completed"})

    def test_interrupt_class(self):
        """Test Interrupt class creation and serialization"""
        from ag_ui.core.events import Interrupt

        interrupt = Interrupt(
            id="int-123",
            reason="database_modification",
            payload={
                "action": "DELETE",
                "table": "users",
                "affectedRows": 42,
            },
        )
        self.assertEqual(interrupt.id, "int-123")
        self.assertEqual(interrupt.reason, "database_modification")
        self.assertEqual(interrupt.payload["affectedRows"], 42)

        # Test serialization uses camelCase
        serialized = interrupt.model_dump(by_alias=True)
        self.assertIn("id", serialized)
        self.assertIn("reason", serialized)
        self.assertIn("payload", serialized)

    def test_interrupt_minimal(self):
        """Test Interrupt with minimal fields"""
        from ag_ui.core.events import Interrupt

        interrupt = Interrupt()
        self.assertIsNone(interrupt.id)
        self.assertIsNone(interrupt.reason)
        self.assertIsNone(interrupt.payload)

    def test_run_finished_serialization_roundtrip(self):
        """Test RunFinishedEvent serialization and deserialization"""
        from ag_ui.core.events import Interrupt

        interrupt = Interrupt(
            id="int-xyz",
            reason="human_approval",
            payload={"approved": True},
        )

        event = RunFinishedEvent(
            thread_id="t1",
            run_id="r1",
            outcome="interrupt",
            interrupt=interrupt,
        )

        # Serialize to JSON
        json_str = event.model_dump_json(by_alias=True)

        # Deserialize
        event_adapter = TypeAdapter(Event)
        deserialized = event_adapter.validate_json(json_str)

        # Verify round-trip
        self.assertEqual(deserialized.thread_id, "t1")
        self.assertEqual(deserialized.run_id, "r1")
        self.assertEqual(deserialized.outcome, "interrupt")
        self.assertEqual(deserialized.interrupt.id, "int-xyz")


if __name__ == "__main__":
    unittest.main()

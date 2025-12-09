"""
Shared pytest fixtures for LangGraph integration tests.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MockChunkContent:
    """Mock content item for LangChain message chunks."""

    thinking: Optional[str] = None
    text: Optional[str] = None
    type: Optional[str] = None
    index: int = 0


@dataclass
class MockAdditionalKwargs:
    """Mock additional_kwargs for OpenAI reasoning format."""

    reasoning: Optional[Dict[str, Any]] = None


@dataclass
class MockChunk:
    """Mock LangChain message chunk for testing reasoning extraction."""

    content: Any = None
    additional_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.additional_kwargs is None:
            self.additional_kwargs = {}


@dataclass
class MockInterruptValue:
    """Mock interrupt value object."""

    reason: str = "human_approval"
    payload: Optional[Dict[str, Any]] = None


@dataclass
class MockInterrupt:
    """Mock LangGraph interrupt object."""

    value: Any = None

    def __post_init__(self):
        if self.value is None:
            self.value = MockInterruptValue()


@dataclass
class MockTask:
    """Mock LangGraph task with interrupts."""

    interrupts: List[MockInterrupt] = None

    def __post_init__(self):
        if self.interrupts is None:
            self.interrupts = []


@dataclass
class MockStateValues:
    """Mock state values from LangGraph."""

    messages: List[Any] = None
    tools: List[Any] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.tools is None:
            self.tools = []


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


# Pytest fixtures


@pytest.fixture
def mock_anthropic_thinking_chunk():
    """Mock LangChain chunk with Anthropic thinking content."""
    return MockChunk(
        content=[{"thinking": "Let me analyze this step by step...", "type": "thinking", "index": 0}]
    )


@pytest.fixture
def mock_anthropic_thinking_chunk_with_index():
    """Factory for mock chunks with specific index."""

    def _create(text: str, index: int = 0):
        return MockChunk(content=[{"thinking": text, "type": "thinking", "index": index}])

    return _create


@pytest.fixture
def mock_openai_reasoning_chunk():
    """Mock LangChain chunk with OpenAI reasoning content."""
    return MockChunk(
        content=[],
        additional_kwargs={
            "reasoning": {"summary": [{"text": "Considering the options...", "index": 0}]}
        },
    )


@pytest.fixture
def mock_empty_chunk():
    """Mock LangChain chunk with no reasoning content."""
    return MockChunk(content=[{"type": "text", "text": "Hello world"}])


@pytest.fixture
def mock_chunk_missing_thinking():
    """Mock chunk with content but no thinking key."""
    return MockChunk(content=[{"type": "text", "text": "Regular text", "index": 0}])


@pytest.fixture
def mock_interrupt():
    """Mock LangGraph interrupt object."""
    return MockInterrupt(value={"reason": "human_approval", "payload": {"action": "approve_email"}})


@pytest.fixture
def mock_interrupt_with_dict_value():
    """Mock interrupt with dict value containing reason."""
    return MockInterrupt(value={"reason": "database_modification", "payload": {"table": "users"}})


@pytest.fixture
def mock_interrupt_with_string_value():
    """Mock interrupt with string value."""
    return MockInterrupt(value="Please approve this action")


@pytest.fixture
def mock_agent_state_with_interrupt():
    """Mock agent state containing active interrupts."""
    interrupt = MockInterrupt(
        value={"reason": "human_approval", "payload": {"tool": "send_email", "args": {"to": "test@example.com"}}}
    )
    task = MockTask(interrupts=[interrupt])
    return MockAgentState(
        values={"messages": [], "tools": []},
        tasks=[task],
        metadata={"writes": {}},
    )


@pytest.fixture
def mock_agent_state_no_interrupt():
    """Mock agent state with no interrupts."""
    return MockAgentState(
        values={"messages": [], "tools": []},
        tasks=[],
        metadata={"writes": {}},
    )

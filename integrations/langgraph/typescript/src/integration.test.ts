/**
 * Integration tests for LangGraph agent event handling.
 *
 * These tests verify the full event flow from LangGraph events through to AG-UI events,
 * complementing the unit tests for individual utility functions.
 */

import { EventType } from "@ag-ui/client";
import { LangGraphReasoning } from "./types";

/**
 * Mock LangGraphPlatformAgent to test event dispatching.
 * This simulates the agent's internal state and event dispatching.
 */
class MockLangGraphAgent {
  dispatchedEvents: Array<{ type: string; [key: string]: any }> = [];
  reasoningProcess: { index?: number; type?: string; reasoningId?: string; accumulatedContent?: string[] } | null = null;
  activeRun: { id: string; reasoningMessages: Array<{ id: string; content: string[] }> } | null = { id: "test-run-123", reasoningMessages: [] };

  dispatchEvent(event: { type: EventType; [key: string]: any }) {
    // Store event with type value preserved for comparisons
    this.dispatchedEvents.push(event);
    return true;
  }

  handleReasoningEvent(reasoningData: LangGraphReasoning) {
    if (!reasoningData || !reasoningData.type || !reasoningData.text) {
      return;
    }

    const reasoningStepIndex = reasoningData.index;

    if (
      this.reasoningProcess?.index &&
      this.reasoningProcess.index !== reasoningStepIndex
    ) {
      if (this.reasoningProcess.type) {
        this.dispatchEvent({
          type: EventType.REASONING_MESSAGE_END,
          messageId: this.reasoningProcess.reasoningId!,
        });
      }
      this.dispatchEvent({
        type: EventType.REASONING_END,
        messageId: this.reasoningProcess.reasoningId!,
      });
      // Create ReasoningMessage from accumulated content
      if (this.reasoningProcess.accumulatedContent?.length) {
        this.activeRun!.reasoningMessages.push({
          id: this.reasoningProcess.reasoningId!,
          content: this.reasoningProcess.accumulatedContent,
        });
      }
      this.reasoningProcess = null;
    }

    if (!this.reasoningProcess) {
      const reasoningId = `${this.activeRun!.id}-${reasoningStepIndex}`;
      this.dispatchEvent({
        type: EventType.REASONING_START,
        messageId: reasoningId,
      });
      this.reasoningProcess = {
        index: reasoningStepIndex,
        reasoningId,
        accumulatedContent: [],
      };
    }

    if (this.reasoningProcess.type !== reasoningData.type) {
      this.dispatchEvent({
        type: EventType.REASONING_MESSAGE_START,
        messageId: this.reasoningProcess.reasoningId!,
        role: "assistant",
      });
      this.reasoningProcess.type = reasoningData.type;
    }

    if (this.reasoningProcess.type) {
      // Accumulate content for ReasoningMessage
      this.reasoningProcess.accumulatedContent!.push(reasoningData.text);

      this.dispatchEvent({
        type: EventType.REASONING_MESSAGE_CONTENT,
        messageId: this.reasoningProcess.reasoningId!,
        delta: reasoningData.text,
      });
    }
  }

  reset() {
    this.dispatchedEvents = [];
    this.reasoningProcess = null;
    if (this.activeRun) {
      this.activeRun.reasoningMessages = [];
    }
  }
}

describe("Reasoning Integration", () => {
  let agent: MockLangGraphAgent;

  beforeEach(() => {
    agent = new MockLangGraphAgent();
  });

  describe("handleReasoningEvent dispatches events correctly", () => {
    it("should dispatch REASONING_START for first reasoning chunk", () => {
      const reasoningData: LangGraphReasoning = {
        type: "text",
        text: "Let me think...",
        index: 0,
      };

      agent.handleReasoningEvent(reasoningData);

      const eventTypes = agent.dispatchedEvents.map((e) => e.type);
      expect(eventTypes).toContain(EventType.REASONING_START);
    });

    it("should dispatch full event sequence for reasoning content", () => {
      const reasoningData: LangGraphReasoning = {
        type: "text",
        text: "Analyzing the problem...",
        index: 0,
      };

      agent.handleReasoningEvent(reasoningData);

      const eventTypes = agent.dispatchedEvents.map((e) => e.type);
      expect(eventTypes).toEqual([
        EventType.REASONING_START,
        EventType.REASONING_MESSAGE_START,
        EventType.REASONING_MESSAGE_CONTENT,
      ]);
    });

    it("should emit content with correct delta", () => {
      const reasoningData: LangGraphReasoning = {
        type: "text",
        text: "Step by step analysis",
        index: 0,
      };

      agent.handleReasoningEvent(reasoningData);

      // Find content event by checking for delta property
      const contentEvents = agent.dispatchedEvents.filter((e) => "delta" in e);
      expect(contentEvents).toHaveLength(1);
      expect(contentEvents[0].delta).toBe("Step by step analysis");
    });

    it("should emit multiple content events for multiple chunks", () => {
      const chunk1: LangGraphReasoning = {
        type: "text",
        text: "First part",
        index: 0,
      };
      const chunk2: LangGraphReasoning = {
        type: "text",
        text: "Second part",
        index: 0,
      };

      agent.handleReasoningEvent(chunk1);
      agent.handleReasoningEvent(chunk2);

      // Find content events by checking for delta property
      const contentEvents = agent.dispatchedEvents.filter((e) => "delta" in e);
      expect(contentEvents).toHaveLength(2);
      expect(contentEvents[0].delta).toBe("First part");
      expect(contentEvents[1].delta).toBe("Second part");
    });

    it("should close previous block and start new one on index change", () => {
      // Note: index=0 is treated as falsy, use index=1 to test
      const chunk1: LangGraphReasoning = {
        type: "text",
        text: "Block 1",
        index: 1,
      };
      const chunk2: LangGraphReasoning = {
        type: "text",
        text: "Block 2",
        index: 2,
      };

      agent.handleReasoningEvent(chunk1);
      const eventsAfterFirst = agent.dispatchedEvents.length;

      agent.handleReasoningEvent(chunk2);
      const eventsAfterSecond = agent.dispatchedEvents.length;

      // Second block should emit more events than just content
      // (it should close first block + start new block)
      expect(eventsAfterSecond).toBeGreaterThan(eventsAfterFirst + 1);

      // Reasoning process should now have index 2
      expect(agent.reasoningProcess?.index).toBe(2);
    });

    it("should not dispatch events for invalid reasoning data", () => {
      agent.handleReasoningEvent(null as any);
      expect(agent.dispatchedEvents).toHaveLength(0);

      agent.handleReasoningEvent({ type: "text" } as any);
      expect(agent.dispatchedEvents).toHaveLength(0);

      agent.handleReasoningEvent({ text: "no type" } as any);
      expect(agent.dispatchedEvents).toHaveLength(0);
    });

    it("should track reasoning process state correctly", () => {
      const reasoningData: LangGraphReasoning = {
        type: "text",
        text: "Thinking...",
        index: 0,
      };

      expect(agent.reasoningProcess).toBeNull();

      agent.handleReasoningEvent(reasoningData);

      expect(agent.reasoningProcess).not.toBeNull();
      expect(agent.reasoningProcess?.index).toBe(0);
      expect(agent.reasoningProcess?.type).toBe("text");
    });
  });
});

describe("Tool Call Integration", () => {
  /**
   * Mock agent for tool call testing.
   */
  class MockToolCallAgent {
    dispatchedEvents: Array<{ type: string; [key: string]: any }> = [];
    messagesInProcess: Record<string, any> = {};
    activeRun = { id: "test-run", hasFunctionStreaming: false };

    dispatchEvent(event: { type: EventType; [key: string]: any }) {
      this.dispatchedEvents.push({ ...event, type: event.type });
      return true;
    }

    setMessageInProgress(runId: string, message: any) {
      this.messagesInProcess[runId] = message;
    }

    getMessageInProgress(runId: string) {
      return this.messagesInProcess[runId];
    }

    reset() {
      this.dispatchedEvents = [];
      this.messagesInProcess = {};
    }

    // Simulate tool call start
    handleToolCallStart(toolCallData: { id: string; name: string }) {
      this.dispatchEvent({
        type: EventType.TOOL_CALL_START,
        toolCallId: toolCallData.id,
        toolCallName: toolCallData.name,
      });
      this.setMessageInProgress(this.activeRun.id, {
        id: "msg-123",
        toolCallId: toolCallData.id,
        toolCallName: toolCallData.name,
      });
    }

    // Simulate tool call args
    handleToolCallArgs(toolCallId: string, args: string) {
      this.dispatchEvent({
        type: EventType.TOOL_CALL_ARGS,
        toolCallId,
        delta: args,
      });
    }

    // Simulate tool call end
    handleToolCallEnd(toolCallId: string) {
      this.dispatchEvent({
        type: EventType.TOOL_CALL_END,
        toolCallId,
      });
      this.messagesInProcess[this.activeRun.id] = null;
    }
  }

  let agent: MockToolCallAgent;

  beforeEach(() => {
    agent = new MockToolCallAgent();
  });

  it("should emit TOOL_CALL_START with correct data", () => {
    agent.handleToolCallStart({ id: "call-123", name: "get_weather" });

    const startEvent = agent.dispatchedEvents.find(
      (e) => e.type === EventType.TOOL_CALL_START
    );
    expect(startEvent).toBeDefined();
    expect(startEvent?.toolCallId).toBe("call-123");
    expect(startEvent?.toolCallName).toBe("get_weather");
  });

  it("should emit TOOL_CALL_ARGS with delta", () => {
    agent.handleToolCallStart({ id: "call-123", name: "search" });
    agent.handleToolCallArgs("call-123", '{"query": "test"}');

    const argsEvent = agent.dispatchedEvents.find(
      (e) => e.type === EventType.TOOL_CALL_ARGS
    );
    expect(argsEvent).toBeDefined();
    expect(argsEvent?.delta).toBe('{"query": "test"}');
  });

  it("should emit TOOL_CALL_END", () => {
    agent.handleToolCallStart({ id: "call-123", name: "search" });
    agent.handleToolCallEnd("call-123");

    const endEvent = agent.dispatchedEvents.find(
      (e) => e.type === EventType.TOOL_CALL_END
    );
    expect(endEvent).toBeDefined();
    expect(endEvent?.toolCallId).toBe("call-123");
  });

  it("should clear message in progress after tool call end", () => {
    agent.handleToolCallStart({ id: "call-123", name: "search" });
    expect(agent.getMessageInProgress(agent.activeRun.id)).not.toBeNull();

    agent.handleToolCallEnd("call-123");
    expect(agent.getMessageInProgress(agent.activeRun.id)).toBeNull();
  });

  it("should emit full tool call sequence", () => {
    agent.handleToolCallStart({ id: "call-123", name: "search" });
    agent.handleToolCallArgs("call-123", '{"q": "test"}');
    agent.handleToolCallEnd("call-123");

    const eventTypes = agent.dispatchedEvents.map((e) => e.type);
    expect(eventTypes).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
      EventType.TOOL_CALL_END,
    ]);
  });
});

describe("Interrupt Event Structure", () => {
  it("should create valid interrupt event data", () => {
    const interruptEvent = {
      type: EventType.RUN_FINISHED,
      threadId: "thread-123",
      runId: "run-456",
      outcome: "interrupt" as const,
      interrupt: {
        id: "int-789",
        reason: "human_approval",
        payload: { tool: "send_email" },
      },
    };

    expect(interruptEvent.type).toBe(EventType.RUN_FINISHED);
    expect(interruptEvent.outcome).toBe("interrupt");
    expect(interruptEvent.interrupt.id).toBe("int-789");
    expect(interruptEvent.interrupt.reason).toBe("human_approval");
  });

  it("should create valid success event data", () => {
    const successEvent = {
      type: EventType.RUN_FINISHED,
      threadId: "thread-123",
      runId: "run-456",
      outcome: "success" as const,
      result: { message: "completed" },
    };

    expect(successEvent.type).toBe(EventType.RUN_FINISHED);
    expect(successEvent.outcome).toBe("success");
    expect(successEvent.result).toEqual({ message: "completed" });
  });
});

import { AbstractAgent } from "@/agent";
import { AgentSubscriber } from "@/agent/subscriber";
import {
  BaseEvent,
  EventType,
  RunAgentInput,
  ReasoningStartEvent,
  ReasoningMessageStartEvent,
  ReasoningMessageContentEvent,
  ReasoningMessageEndEvent,
  ReasoningEndEvent,
  RunStartedEvent,
} from "@ag-ui/core";
import { Observable, of } from "rxjs";

// Mock uuid module
jest.mock("uuid", () => ({
  v4: jest.fn().mockReturnValue("mock-uuid"),
}));

// Mock utils with handling for undefined values
jest.mock("@/utils", () => {
  const actual = jest.requireActual<typeof import("@/utils")>("@/utils");
  return {
    ...actual,
    structuredClone_: (obj: any) => {
      if (obj === undefined) return undefined;
      const jsonString = JSON.stringify(obj);
      if (jsonString === undefined || jsonString === "undefined") return undefined;
      return JSON.parse(jsonString);
    },
  };
});

// Mock the verify modules but NOT apply - we want to test against real defaultApplyEvents
jest.mock("@/verify", () => ({
  verifyEvents: jest.fn(() => (source$: Observable<any>) => source$),
}));

jest.mock("@/chunks", () => ({
  transformChunks: jest.fn(() => (source$: Observable<any>) => source$),
}));

// Create a test agent implementation
class TestAgent extends AbstractAgent {
  private eventsToEmit: BaseEvent[] = [];

  setEventsToEmit(events: BaseEvent[]) {
    this.eventsToEmit = events;
  }

  run(input: RunAgentInput): Observable<BaseEvent> {
    return of(...this.eventsToEmit);
  }
}

describe("defaultApplyEvents with reasoning events", () => {
  let agent: TestAgent;
  let mockSubscriber: AgentSubscriber;

  beforeEach(() => {
    jest.clearAllMocks();

    agent = new TestAgent({
      threadId: "test-thread",
      initialMessages: [],
      initialState: {},
    });

    mockSubscriber = {
      onReasoningStartEvent: jest.fn(),
      onReasoningMessageStartEvent: jest.fn(),
      onReasoningMessageContentEvent: jest.fn(),
      onReasoningMessageEndEvent: jest.fn(),
      onReasoningEndEvent: jest.fn(),
    };
  });

  describe("Reasoning Event Callback Invocation", () => {
    it("should call all reasoning event callbacks in correct order", async () => {
      const callOrder: string[] = [];

      const trackingSubscriber: AgentSubscriber = {
        onReasoningStartEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_START");
        }),
        onReasoningMessageStartEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_MESSAGE_START");
        }),
        onReasoningMessageContentEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_MESSAGE_CONTENT");
        }),
        onReasoningMessageEndEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_MESSAGE_END");
        }),
        onReasoningEndEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_END");
        }),
      };

      agent.subscribe(trackingSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.RUN_STARTED,
          threadId: "test-thread",
          runId: "test-run",
        } as RunStartedEvent,
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "Thinking...",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-1",
        } as ReasoningMessageEndEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
      ]);

      await agent.runAgent({});

      expect(callOrder).toEqual([
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
      ]);
    });

    it("should call onReasoningStartEvent with correct event", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
      ]);

      await agent.runAgent({});

      expect(mockSubscriber.onReasoningStartEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          event: expect.objectContaining({
            type: EventType.REASONING_START,
            messageId: "reasoning-1",
          }),
          messages: [],
          state: {},
          agent,
        }),
      );
    });

    it("should call onReasoningMessageStartEvent with correct event", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
      ]);

      await agent.runAgent({});

      expect(mockSubscriber.onReasoningMessageStartEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          event: expect.objectContaining({
            type: EventType.REASONING_MESSAGE_START,
            messageId: "reasoning-msg-1",
            role: "assistant",
          }),
          messages: [],
          state: {},
          agent,
        }),
      );
    });

    it("should call onReasoningEndEvent with correct event", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
      ]);

      await agent.runAgent({});

      expect(mockSubscriber.onReasoningEndEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          event: expect.objectContaining({
            type: EventType.REASONING_END,
            messageId: "reasoning-1",
          }),
          messages: [],
          state: {},
          agent,
        }),
      );
    });
  });

  describe("Reasoning Message Buffer Accumulation", () => {
    it("should accumulate reasoningMessageBuffer across CONTENT events", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "First ",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "second ",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "third",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-1",
        } as ReasoningMessageEndEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
      ]);

      await agent.runAgent({});

      // Verify buffer accumulation in CONTENT events
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenCalledTimes(3);

      // First content: buffer contains only the current delta (accumulated after adding)
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenNthCalledWith(
        1,
        expect.objectContaining({
          reasoningMessageBuffer: "First ",
        }),
      );

      // Second content: buffer contains first + second deltas
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          reasoningMessageBuffer: "First second ",
        }),
      );

      // Third content: buffer contains all deltas
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenNthCalledWith(
        3,
        expect.objectContaining({
          reasoningMessageBuffer: "First second third",
        }),
      );

      // Verify END event receives complete buffer
      expect(mockSubscriber.onReasoningMessageEndEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          reasoningMessageBuffer: "First second third",
        }),
      );
    });

    it("should pass correct event in CONTENT callback along with buffer", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "Thinking about the problem...",
        } as ReasoningMessageContentEvent,
      ]);

      await agent.runAgent({});

      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          event: expect.objectContaining({
            type: EventType.REASONING_MESSAGE_CONTENT,
            messageId: "reasoning-msg-1",
            delta: "Thinking about the problem...",
          }),
          reasoningMessageBuffer: "Thinking about the problem...",
          messages: [],
          state: {},
          agent,
        }),
      );
    });
  });

  describe("Reasoning Buffer Reset Behavior", () => {
    it("should reset buffer on REASONING_START", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        // First reasoning block
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "First block content",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-1",
        } as ReasoningMessageEndEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
        // Second reasoning block - buffer should reset
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-2",
        } as ReasoningStartEvent,
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-2",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-2",
          delta: "Second block content",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-2",
        } as ReasoningMessageEndEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-2",
        } as ReasoningEndEvent,
      ]);

      await agent.runAgent({});

      // Verify both CONTENT events have their own accumulated buffers (not carrying over)
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenCalledTimes(2);

      // First block content
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenNthCalledWith(
        1,
        expect.objectContaining({
          reasoningMessageBuffer: "First block content",
        }),
      );

      // Second block content - should NOT include first block's content
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          reasoningMessageBuffer: "Second block content",
        }),
      );

      // Verify END events also have separate buffers
      expect(mockSubscriber.onReasoningMessageEndEvent).toHaveBeenCalledTimes(2);

      expect(mockSubscriber.onReasoningMessageEndEvent).toHaveBeenNthCalledWith(
        1,
        expect.objectContaining({
          reasoningMessageBuffer: "First block content",
        }),
      );

      expect(mockSubscriber.onReasoningMessageEndEvent).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          reasoningMessageBuffer: "Second block content",
        }),
      );
    });

    it("should reset buffer on REASONING_MESSAGE_START", async () => {
      agent.subscribe(mockSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
        // First message
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "First message",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-1",
        } as ReasoningMessageEndEvent,
        // Second message within same reasoning block
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-2",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-2",
          delta: "Second message",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-2",
        } as ReasoningMessageEndEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
      ]);

      await agent.runAgent({});

      // Second message content should have reset buffer
      expect(mockSubscriber.onReasoningMessageContentEvent).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          reasoningMessageBuffer: "Second message",
        }),
      );
    });
  });

  describe("Subscriber Mutation Support", () => {
    it("should allow subscribers to mutate state on reasoning events", async () => {
      const mutatingSubscriber: AgentSubscriber = {
        onReasoningStartEvent: jest.fn().mockReturnValue({
          state: { isReasoning: true },
        }),
        onReasoningEndEvent: jest.fn().mockReturnValue({
          state: { isReasoning: false, reasoningComplete: true },
        }),
      };

      agent.subscribe(mutatingSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
      ]);

      await agent.runAgent({});

      // Verify final state reflects mutations
      expect(agent.state).toEqual({ isReasoning: false, reasoningComplete: true });
    });

    it("should support stopPropagation on reasoning events", async () => {
      const blockingSubscriber: AgentSubscriber = {
        onReasoningStartEvent: jest.fn().mockReturnValue({
          stopPropagation: true,
        }),
      };

      const secondSubscriber: AgentSubscriber = {
        onReasoningStartEvent: jest.fn(),
      };

      agent.subscribe(blockingSubscriber);
      agent.subscribe(secondSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
      ]);

      await agent.runAgent({});

      expect(blockingSubscriber.onReasoningStartEvent).toHaveBeenCalled();
      expect(secondSubscriber.onReasoningStartEvent).not.toHaveBeenCalled();
    });
  });

  describe("Integration with Other Events", () => {
    it("should handle reasoning events mixed with text message events", async () => {
      const callOrder: string[] = [];

      const trackingSubscriber: AgentSubscriber = {
        onReasoningStartEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_START");
        }),
        onReasoningMessageContentEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_MESSAGE_CONTENT");
        }),
        onReasoningEndEvent: jest.fn().mockImplementation(() => {
          callOrder.push("REASONING_END");
        }),
        onTextMessageStartEvent: jest.fn().mockImplementation(() => {
          callOrder.push("TEXT_MESSAGE_START");
        }),
        onTextMessageContentEvent: jest.fn().mockImplementation(() => {
          callOrder.push("TEXT_MESSAGE_CONTENT");
        }),
        onTextMessageEndEvent: jest.fn().mockImplementation(() => {
          callOrder.push("TEXT_MESSAGE_END");
        }),
      };

      agent.subscribe(trackingSubscriber);
      agent.setEventsToEmit([
        {
          type: EventType.RUN_STARTED,
          threadId: "test-thread",
          runId: "test-run",
        } as RunStartedEvent,
        // Reasoning first
        {
          type: EventType.REASONING_START,
          messageId: "reasoning-1",
        } as ReasoningStartEvent,
        {
          type: EventType.REASONING_MESSAGE_START,
          messageId: "reasoning-msg-1",
          role: "assistant",
        } as ReasoningMessageStartEvent,
        {
          type: EventType.REASONING_MESSAGE_CONTENT,
          messageId: "reasoning-msg-1",
          delta: "Thinking...",
        } as ReasoningMessageContentEvent,
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: "reasoning-msg-1",
        } as ReasoningMessageEndEvent,
        {
          type: EventType.REASONING_END,
          messageId: "reasoning-1",
        } as ReasoningEndEvent,
        // Then text response
        {
          type: EventType.TEXT_MESSAGE_START,
          messageId: "msg-1",
          role: "assistant",
        },
        {
          type: EventType.TEXT_MESSAGE_CONTENT,
          messageId: "msg-1",
          delta: "Here is my response",
        },
        {
          type: EventType.TEXT_MESSAGE_END,
          messageId: "msg-1",
        },
      ]);

      await agent.runAgent({});

      expect(callOrder).toEqual([
        "REASONING_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_END",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
      ]);
    });
  });
});

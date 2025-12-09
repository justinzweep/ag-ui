/**
 * Tests for interrupt detection and resume handling in LangGraph integration.
 */

import {
  EventType,
  RunFinishedEvent,
  RunAgentInput,
  Interrupt,
  Resume,
} from "@ag-ui/client";

describe("Interrupt Handling", () => {
  describe("RunFinishedEvent with interrupt outcome", () => {
    it("should create valid RunFinishedEvent with interrupt outcome", () => {
      const event: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-1",
        runId: "run-1",
        outcome: "interrupt",
        interrupt: {
          id: "int-123",
          reason: "human_approval",
          payload: { tool: "send_email" },
        },
      };

      expect(event.type).toBe(EventType.RUN_FINISHED);
      expect(event.outcome).toBe("interrupt");
      expect(event.interrupt?.id).toBe("int-123");
      expect(event.interrupt?.reason).toBe("human_approval");
      expect(event.interrupt?.payload).toEqual({ tool: "send_email" });
    });

    it("should create valid RunFinishedEvent with success outcome", () => {
      const event: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-1",
        runId: "run-1",
        outcome: "success",
        result: { message: "completed" },
      };

      expect(event.type).toBe(EventType.RUN_FINISHED);
      expect(event.outcome).toBe("success");
      expect(event.result).toEqual({ message: "completed" });
    });

    it("should allow RunFinishedEvent without outcome for backward compatibility", () => {
      const event: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-1",
        runId: "run-1",
        result: { data: "legacy" },
      };

      expect(event.type).toBe(EventType.RUN_FINISHED);
      expect(event.outcome).toBeUndefined();
    });

    it("should allow minimal interrupt object", () => {
      const event: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-1",
        runId: "run-1",
        outcome: "interrupt",
        interrupt: {},
      };

      expect(event.interrupt).toEqual({});
    });
  });

  describe("Interrupt object", () => {
    it("should support complete interrupt object", () => {
      const interrupt: Interrupt = {
        id: "int-123",
        reason: "database_modification",
        payload: {
          action: "DELETE",
          table: "users",
          affectedRows: 42,
        },
      };

      expect(interrupt.id).toBe("int-123");
      expect(interrupt.reason).toBe("database_modification");
      expect(interrupt.payload?.affectedRows).toBe(42);
    });

    it("should support interrupt with only id", () => {
      const interrupt: Interrupt = {
        id: "int-456",
      };

      expect(interrupt.id).toBe("int-456");
      expect(interrupt.reason).toBeUndefined();
      expect(interrupt.payload).toBeUndefined();
    });

    it("should support empty interrupt object", () => {
      const interrupt: Interrupt = {};

      expect(interrupt.id).toBeUndefined();
      expect(interrupt.reason).toBeUndefined();
      expect(interrupt.payload).toBeUndefined();
    });
  });

  describe("Resume handling", () => {
    it("should support resume in RunAgentInput", () => {
      const input: RunAgentInput = {
        threadId: "thread-1",
        runId: "run-2",
        state: {},
        messages: [],
        tools: [],
        context: [],
        forwardedProps: {},
        resume: {
          interruptId: "int-abc123",
          payload: { approved: true },
        },
      };

      expect(input.resume?.interruptId).toBe("int-abc123");
      expect(input.resume?.payload).toEqual({ approved: true });
    });

    it("should support RunAgentInput without resume for backward compatibility", () => {
      const input: RunAgentInput = {
        threadId: "thread-1",
        runId: "run-1",
        state: {},
        messages: [],
        tools: [],
        context: [],
        forwardedProps: {},
      };

      expect(input.resume).toBeUndefined();
    });

    it("should support resume with only payload", () => {
      const resume: Resume = {
        payload: { approved: false },
      };

      expect(resume.interruptId).toBeUndefined();
      expect(resume.payload).toEqual({ approved: false });
    });

    it("should support empty resume object", () => {
      const resume: Resume = {};

      expect(resume.interruptId).toBeUndefined();
      expect(resume.payload).toBeUndefined();
    });
  });

  describe("Complete interrupt/resume flow", () => {
    it("should validate complete interrupt to resume flow", () => {
      // Step 1: Agent sends interrupt
      const interruptEvent: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-1",
        runId: "run-1",
        outcome: "interrupt",
        interrupt: {
          id: "int-abc123",
          reason: "human_approval",
          payload: {
            tool: "send_email",
            args: { to: "test@example.com" },
          },
        },
      };

      expect(interruptEvent.outcome).toBe("interrupt");
      expect(interruptEvent.interrupt).toBeDefined();

      // Step 2: User responds with approval
      const resumeInput: RunAgentInput = {
        threadId: "thread-1",
        runId: "run-2",
        state: {},
        messages: [],
        tools: [],
        context: [],
        forwardedProps: {},
        resume: {
          interruptId: "int-abc123",
          payload: { approved: true },
        },
      };

      expect(resumeInput.threadId).toBe(interruptEvent.threadId);
      expect(resumeInput.resume?.interruptId).toBe(interruptEvent.interrupt?.id);

      // Step 3: Agent completes successfully
      const successEvent: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-1",
        runId: "run-2",
        outcome: "success",
        result: { emailSent: true },
      };

      expect(successEvent.outcome).toBe("success");
      expect(successEvent.threadId).toBe(resumeInput.threadId);
    });

    it("should support complex approval flow with modifications", () => {
      // Agent requests approval with context
      const interruptEvent: RunFinishedEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "thread-456",
        runId: "run-789",
        outcome: "interrupt",
        interrupt: {
          id: "approval-001",
          reason: "database_modification",
          payload: {
            action: "DELETE",
            table: "users",
            affectedRows: 42,
            query: "DELETE FROM users WHERE last_login < '2023-01-01'",
            riskLevel: "high",
          },
        },
      };

      expect(interruptEvent.interrupt?.payload?.riskLevel).toBe("high");

      // User approves with modifications
      const resumeInput: RunAgentInput = {
        threadId: "thread-456",
        runId: "run-790",
        state: {},
        messages: [],
        tools: [],
        context: [],
        forwardedProps: {},
        resume: {
          interruptId: "approval-001",
          payload: {
            approved: true,
            modifications: {
              batchSize: 10,
              dryRun: true,
            },
          },
        },
      };

      expect(resumeInput.resume?.payload?.modifications?.dryRun).toBe(true);
    });
  });

  describe("Interrupt detection helpers", () => {
    it("should extract reason from interrupt value dict", () => {
      const interrupt = {
        value: { reason: "human_approval", payload: {} },
      };

      const reason =
        interrupt.value && typeof interrupt.value === "object"
          ? (interrupt.value as any).reason || "human_approval"
          : "human_approval";

      expect(reason).toBe("human_approval");
    });

    it("should default to human_approval when reason is missing", () => {
      const interrupt = {
        value: { payload: "some data" },
      };

      const reason =
        interrupt.value && typeof interrupt.value === "object"
          ? (interrupt.value as any).reason || "human_approval"
          : "human_approval";

      expect(reason).toBe("human_approval");
    });

    it("should default to human_approval for non-object interrupt value", () => {
      const interrupt = {
        value: "string interrupt value",
      };

      const reason =
        interrupt.value && typeof interrupt.value === "object"
          ? (interrupt.value as any).reason || "human_approval"
          : "human_approval";

      expect(reason).toBe("human_approval");
    });
  });
});

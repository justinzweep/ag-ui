import {
  RunFinishedEventSchema,
  RunAgentInputSchema,
  InterruptSchema,
  ResumeSchema,
  EventType,
} from "../index";

describe("Interrupt-Aware Run Lifecycle", () => {
  describe("RunFinishedEvent Schema", () => {
    it("should accept RunFinishedEvent with outcome: success and result", () => {
      const event = {
        type: EventType.RUN_FINISHED,
        threadId: "thread_1",
        runId: "run_1",
        outcome: "success" as const,
        result: { data: "completed" },
      };

      const result = RunFinishedEventSchema.safeParse(event);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe(EventType.RUN_FINISHED);
        expect(result.data.threadId).toBe("thread_1");
        expect(result.data.runId).toBe("run_1");
        expect(result.data.outcome).toBe("success");
        expect(result.data.result).toEqual({ data: "completed" });
        expect(result.data.interrupt).toBeUndefined();
      }
    });

    it("should accept RunFinishedEvent with outcome: interrupt and interrupt object", () => {
      const event = {
        type: EventType.RUN_FINISHED,
        threadId: "thread_1",
        runId: "run_1",
        outcome: "interrupt" as const,
        interrupt: {
          id: "int-abc123",
          reason: "human_approval",
          payload: {
            proposal: {
              tool: "sendEmail",
              args: { to: "a@b.com", subject: "Hi" },
            },
          },
        },
      };

      const result = RunFinishedEventSchema.safeParse(event);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe(EventType.RUN_FINISHED);
        expect(result.data.outcome).toBe("interrupt");
        expect(result.data.interrupt?.id).toBe("int-abc123");
        expect(result.data.interrupt?.reason).toBe("human_approval");
        expect(result.data.interrupt?.payload).toEqual({
          proposal: {
            tool: "sendEmail",
            args: { to: "a@b.com", subject: "Hi" },
          },
        });
        expect(result.data.result).toBeUndefined();
      }
    });

    it("should accept RunFinishedEvent without outcome (backward compatibility)", () => {
      const event = {
        type: EventType.RUN_FINISHED,
        threadId: "thread_1",
        runId: "run_1",
        result: { data: "completed" },
      };

      const result = RunFinishedEventSchema.safeParse(event);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.outcome).toBeUndefined();
        expect(result.data.interrupt).toBeUndefined();
      }
    });

    it("should accept RunFinishedEvent with minimal interrupt object", () => {
      const event = {
        type: EventType.RUN_FINISHED,
        threadId: "thread_1",
        runId: "run_1",
        outcome: "interrupt" as const,
        interrupt: {},
      };

      const result = RunFinishedEventSchema.safeParse(event);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.interrupt).toEqual({});
      }
    });

    it("should reject invalid outcome values", () => {
      const event = {
        type: EventType.RUN_FINISHED,
        threadId: "thread_1",
        runId: "run_1",
        outcome: "invalid_outcome",
      };

      const result = RunFinishedEventSchema.safeParse(event);

      expect(result.success).toBe(false);
    });
  });

  describe("Interrupt Schema", () => {
    it("should accept complete interrupt object", () => {
      const interrupt = {
        id: "int-123",
        reason: "database_modification",
        payload: {
          action: "DELETE",
          table: "users",
          affectedRows: 42,
        },
      };

      const result = InterruptSchema.safeParse(interrupt);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.id).toBe("int-123");
        expect(result.data.reason).toBe("database_modification");
        expect(result.data.payload).toEqual({
          action: "DELETE",
          table: "users",
          affectedRows: 42,
        });
      }
    });

    it("should accept empty interrupt object", () => {
      const interrupt = {};

      const result = InterruptSchema.safeParse(interrupt);

      expect(result.success).toBe(true);
    });

    it("should accept interrupt with only id", () => {
      const interrupt = { id: "int-456" };

      const result = InterruptSchema.safeParse(interrupt);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.id).toBe("int-456");
        expect(result.data.reason).toBeUndefined();
        expect(result.data.payload).toBeUndefined();
      }
    });
  });

  describe("Resume Schema", () => {
    it("should accept complete resume object", () => {
      const resume = {
        interruptId: "int-abc123",
        payload: { approved: true, modifications: { batchSize: 10 } },
      };

      const result = ResumeSchema.safeParse(resume);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.interruptId).toBe("int-abc123");
        expect(result.data.payload).toEqual({
          approved: true,
          modifications: { batchSize: 10 },
        });
      }
    });

    it("should accept empty resume object", () => {
      const resume = {};

      const result = ResumeSchema.safeParse(resume);

      expect(result.success).toBe(true);
    });

    it("should accept resume with only payload", () => {
      const resume = { payload: { approved: false } };

      const result = ResumeSchema.safeParse(resume);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.interruptId).toBeUndefined();
        expect(result.data.payload).toEqual({ approved: false });
      }
    });
  });

  describe("RunAgentInput Schema with Resume", () => {
    it("should accept RunAgentInput with resume field", () => {
      const input = {
        threadId: "thread_1",
        runId: "run_2",
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

      const result = RunAgentInputSchema.safeParse(input);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.resume?.interruptId).toBe("int-abc123");
        expect(result.data.resume?.payload).toEqual({ approved: true });
      }
    });

    it("should accept RunAgentInput without resume (backward compatibility)", () => {
      const input = {
        threadId: "thread_1",
        runId: "run_1",
        state: {},
        messages: [],
        tools: [],
        context: [],
        forwardedProps: {},
      };

      const result = RunAgentInputSchema.safeParse(input);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.resume).toBeUndefined();
      }
    });

    it("should accept RunAgentInput with empty resume object", () => {
      const input = {
        threadId: "thread_1",
        runId: "run_1",
        state: {},
        messages: [],
        tools: [],
        context: [],
        forwardedProps: {},
        resume: {},
      };

      const result = RunAgentInputSchema.safeParse(input);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.resume).toEqual({});
      }
    });
  });

  describe("Complex interrupt/resume flow", () => {
    it("should validate a complete interrupt scenario", () => {
      // Agent sends interrupt
      const interruptEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "t1",
        runId: "r1",
        outcome: "interrupt" as const,
        interrupt: {
          id: "int-abc123",
          reason: "human_approval",
          payload: {
            proposal: {
              tool: "sendEmail",
              args: { to: "a@b.com", subject: "Hi", body: "..." },
            },
          },
        },
      };

      const interruptResult = RunFinishedEventSchema.safeParse(interruptEvent);
      expect(interruptResult.success).toBe(true);

      // User responds with approval
      const resumeInput = {
        threadId: "t1",
        runId: "r2",
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

      const resumeResult = RunAgentInputSchema.safeParse(resumeInput);
      expect(resumeResult.success).toBe(true);

      // Agent completes successfully
      const successEvent = {
        type: EventType.RUN_FINISHED,
        threadId: "t1",
        runId: "r2",
        outcome: "success" as const,
        result: { emailSent: true },
      };

      const successResult = RunFinishedEventSchema.safeParse(successEvent);
      expect(successResult.success).toBe(true);
    });
  });
});

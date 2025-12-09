import { z } from "zod";
import { MessageSchema, StateSchema, RunAgentInputSchema } from "./types";

// Text messages can have any role except "tool"
const TextMessageRoleSchema = z.union([
  z.literal("developer"),
  z.literal("system"),
  z.literal("assistant"),
  z.literal("user"),
]);

export enum EventType {
  TEXT_MESSAGE_START = "TEXT_MESSAGE_START",
  TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT",
  TEXT_MESSAGE_END = "TEXT_MESSAGE_END",
  TEXT_MESSAGE_CHUNK = "TEXT_MESSAGE_CHUNK",
  REASONING_START = "REASONING_START",
  REASONING_MESSAGE_START = "REASONING_MESSAGE_START",
  REASONING_MESSAGE_CONTENT = "REASONING_MESSAGE_CONTENT",
  REASONING_MESSAGE_END = "REASONING_MESSAGE_END",
  REASONING_MESSAGE_CHUNK = "REASONING_MESSAGE_CHUNK",
  REASONING_END = "REASONING_END",
  TOOL_CALL_START = "TOOL_CALL_START",
  TOOL_CALL_ARGS = "TOOL_CALL_ARGS",
  TOOL_CALL_END = "TOOL_CALL_END",
  TOOL_CALL_CHUNK = "TOOL_CALL_CHUNK",
  TOOL_CALL_RESULT = "TOOL_CALL_RESULT",
  STATE_SNAPSHOT = "STATE_SNAPSHOT",
  STATE_DELTA = "STATE_DELTA",
  MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT",
  ACTIVITY_SNAPSHOT = "ACTIVITY_SNAPSHOT",
  ACTIVITY_DELTA = "ACTIVITY_DELTA",
  RAW = "RAW",
  CUSTOM = "CUSTOM",
  RUN_STARTED = "RUN_STARTED",
  RUN_FINISHED = "RUN_FINISHED",
  RUN_ERROR = "RUN_ERROR",
  STEP_STARTED = "STEP_STARTED",
  STEP_FINISHED = "STEP_FINISHED",
}

export const BaseEventSchema = z.object({
  type: z.nativeEnum(EventType),
  timestamp: z.number().optional(),
  rawEvent: z.any().optional(),
});

export const TextMessageStartEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TEXT_MESSAGE_START),
  messageId: z.string(),
  role: TextMessageRoleSchema.default("assistant"),
});

export const TextMessageContentEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TEXT_MESSAGE_CONTENT),
  messageId: z.string(),
  delta: z.string().refine((s) => s.length > 0, "Delta must not be an empty string"),
});

export const TextMessageEndEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TEXT_MESSAGE_END),
  messageId: z.string(),
});

export const TextMessageChunkEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TEXT_MESSAGE_CHUNK),
  messageId: z.string().optional(),
  role: TextMessageRoleSchema.optional(),
  delta: z.string().optional(),
});

export const ReasoningStartEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.REASONING_START),
  messageId: z.string(),
  encryptedContent: z.string().optional(),
});

export const ReasoningMessageStartEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.REASONING_MESSAGE_START),
  messageId: z.string(),
  role: z.literal("assistant"),
});

export const ReasoningMessageContentEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.REASONING_MESSAGE_CONTENT),
  messageId: z.string(),
  delta: z.string().refine((s) => s.length > 0, "Delta must not be an empty string"),
});

export const ReasoningMessageEndEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.REASONING_MESSAGE_END),
  messageId: z.string(),
});

export const ReasoningMessageChunkEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.REASONING_MESSAGE_CHUNK),
  messageId: z.string().optional(),
  delta: z.string().optional(),
});

export const ReasoningEndEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.REASONING_END),
  messageId: z.string(),
});

export const ToolCallStartEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TOOL_CALL_START),
  toolCallId: z.string(),
  toolCallName: z.string(),
  parentMessageId: z.string().optional(),
});

export const ToolCallArgsEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TOOL_CALL_ARGS),
  toolCallId: z.string(),
  delta: z.string(),
});

export const ToolCallEndEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TOOL_CALL_END),
  toolCallId: z.string(),
});

export const ToolCallResultEventSchema = BaseEventSchema.extend({
  messageId: z.string(),
  type: z.literal(EventType.TOOL_CALL_RESULT),
  toolCallId: z.string(),
  content: z.string(),
  role: z.literal("tool").optional(),
});

export const ToolCallChunkEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.TOOL_CALL_CHUNK),
  toolCallId: z.string().optional(),
  toolCallName: z.string().optional(),
  parentMessageId: z.string().optional(),
  delta: z.string().optional(),
});

export const StateSnapshotEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.STATE_SNAPSHOT),
  snapshot: StateSchema,
});

export const StateDeltaEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.STATE_DELTA),
  delta: z.array(z.any()), // JSON Patch (RFC 6902)
});

export const MessagesSnapshotEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.MESSAGES_SNAPSHOT),
  messages: z.array(MessageSchema),
});

export const ActivitySnapshotEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.ACTIVITY_SNAPSHOT),
  messageId: z.string(),
  activityType: z.string(),
  content: z.record(z.any()),
  replace: z.boolean().optional().default(true),
});

export const ActivityDeltaEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.ACTIVITY_DELTA),
  messageId: z.string(),
  activityType: z.string(),
  patch: z.array(z.any()),
});

export const RawEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.RAW),
  event: z.any(),
  source: z.string().optional(),
});

export const CustomEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.CUSTOM),
  name: z.string(),
  value: z.any(),
});

export const RunStartedEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.RUN_STARTED),
  threadId: z.string(),
  runId: z.string(),
  parentRunId: z.string().optional(),
  input: RunAgentInputSchema.optional(),
});

export const RunFinishedEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.RUN_FINISHED),
  threadId: z.string(),
  runId: z.string(),
  result: z.any().optional(),
});

export const RunErrorEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.RUN_ERROR),
  message: z.string(),
  code: z.string().optional(),
});

export const StepStartedEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.STEP_STARTED),
  stepName: z.string(),
});

export const StepFinishedEventSchema = BaseEventSchema.extend({
  type: z.literal(EventType.STEP_FINISHED),
  stepName: z.string(),
});

export const EventSchemas = z.discriminatedUnion("type", [
  TextMessageStartEventSchema,
  TextMessageContentEventSchema,
  TextMessageEndEventSchema,
  TextMessageChunkEventSchema,
  ReasoningStartEventSchema,
  ReasoningMessageStartEventSchema,
  ReasoningMessageContentEventSchema,
  ReasoningMessageEndEventSchema,
  ReasoningMessageChunkEventSchema,
  ReasoningEndEventSchema,
  ToolCallStartEventSchema,
  ToolCallArgsEventSchema,
  ToolCallEndEventSchema,
  ToolCallChunkEventSchema,
  ToolCallResultEventSchema,
  StateSnapshotEventSchema,
  StateDeltaEventSchema,
  MessagesSnapshotEventSchema,
  ActivitySnapshotEventSchema,
  ActivityDeltaEventSchema,
  RawEventSchema,
  CustomEventSchema,
  RunStartedEventSchema,
  RunFinishedEventSchema,
  RunErrorEventSchema,
  StepStartedEventSchema,
  StepFinishedEventSchema,
]);

export type BaseEvent = z.infer<typeof BaseEventSchema>;
export type TextMessageStartEvent = z.infer<typeof TextMessageStartEventSchema>;
export type TextMessageContentEvent = z.infer<typeof TextMessageContentEventSchema>;
export type TextMessageEndEvent = z.infer<typeof TextMessageEndEventSchema>;
export type TextMessageChunkEvent = z.infer<typeof TextMessageChunkEventSchema>;
export type ReasoningStartEvent = z.infer<typeof ReasoningStartEventSchema>;
export type ReasoningMessageStartEvent = z.infer<typeof ReasoningMessageStartEventSchema>;
export type ReasoningMessageContentEvent = z.infer<typeof ReasoningMessageContentEventSchema>;
export type ReasoningMessageEndEvent = z.infer<typeof ReasoningMessageEndEventSchema>;
export type ReasoningMessageChunkEvent = z.infer<typeof ReasoningMessageChunkEventSchema>;
export type ReasoningEndEvent = z.infer<typeof ReasoningEndEventSchema>;
export type ToolCallStartEvent = z.infer<typeof ToolCallStartEventSchema>;
export type ToolCallArgsEvent = z.infer<typeof ToolCallArgsEventSchema>;
export type ToolCallEndEvent = z.infer<typeof ToolCallEndEventSchema>;
export type ToolCallChunkEvent = z.infer<typeof ToolCallChunkEventSchema>;
export type ToolCallResultEvent = z.infer<typeof ToolCallResultEventSchema>;
export type StateSnapshotEvent = z.infer<typeof StateSnapshotEventSchema>;
export type StateDeltaEvent = z.infer<typeof StateDeltaEventSchema>;
export type MessagesSnapshotEvent = z.infer<typeof MessagesSnapshotEventSchema>;
export type ActivitySnapshotEvent = z.infer<typeof ActivitySnapshotEventSchema>;
export type ActivityDeltaEvent = z.infer<typeof ActivityDeltaEventSchema>;
export type RawEvent = z.infer<typeof RawEventSchema>;
export type CustomEvent = z.infer<typeof CustomEventSchema>;
export type RunStartedEvent = z.infer<typeof RunStartedEventSchema>;
export type RunFinishedEvent = z.infer<typeof RunFinishedEventSchema>;
export type RunErrorEvent = z.infer<typeof RunErrorEventSchema>;
export type StepStartedEvent = z.infer<typeof StepStartedEventSchema>;
export type StepFinishedEvent = z.infer<typeof StepFinishedEventSchema>;

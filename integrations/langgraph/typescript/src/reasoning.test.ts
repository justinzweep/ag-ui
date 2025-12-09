/**
 * Tests for reasoning content extraction.
 */

import { resolveReasoningContent } from "./utils";

describe("resolveReasoningContent", () => {
  describe("Anthropic thinking format", () => {
    it("should extract thinking content from Anthropic format", () => {
      const eventData = {
        chunk: {
          content: [
            { thinking: "Let me analyze this step by step...", type: "thinking", index: 0 },
          ],
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).not.toBeNull();
      expect(result?.text).toBe("Let me analyze this step by step...");
      expect(result?.type).toBe("text");
      expect(result?.index).toBe(0);
    });

    it("should correctly extract index from Anthropic thinking content", () => {
      const eventData = {
        chunk: {
          content: [{ thinking: "Second thought...", type: "thinking", index: 1 }],
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).not.toBeNull();
      expect(result?.index).toBe(1);
    });

    it("should return null when content is empty array", () => {
      const eventData = {
        chunk: {
          content: [],
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should return null when content is undefined", () => {
      const eventData = {
        chunk: {},
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should return null when thinking key is missing", () => {
      const eventData = {
        chunk: {
          content: [{ type: "text", text: "Regular content", index: 0 }],
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should return null when thinking value is empty string", () => {
      const eventData = {
        chunk: {
          content: [{ thinking: "", type: "thinking", index: 0 }],
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should return null when content is not an array", () => {
      const eventData = {
        chunk: {
          content: "string content",
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });
  });

  describe("OpenAI reasoning format", () => {
    it("should extract reasoning from OpenAI format", () => {
      // OpenAI format: content is a string (not array), reasoning in additional_kwargs
      const eventData = {
        chunk: {
          content: "text response",
          additional_kwargs: {
            reasoning: {
              summary: [{ text: "Considering options...", index: 0 }],
            },
          },
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).not.toBeNull();
      expect(result?.text).toBe("Considering options...");
      expect(result?.type).toBe("text");
      expect(result?.index).toBe(0);
    });

    it("should correctly extract index from OpenAI reasoning", () => {
      const eventData = {
        chunk: {
          content: "text response",
          additional_kwargs: {
            reasoning: {
              summary: [{ text: "Step 2...", index: 2 }],
            },
          },
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).not.toBeNull();
      expect(result?.index).toBe(2);
    });

    it("should return null when OpenAI summary is empty", () => {
      const eventData = {
        chunk: {
          content: "text response",
          additional_kwargs: {
            reasoning: {
              summary: [],
            },
          },
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should return null when OpenAI summary item has no text", () => {
      const eventData = {
        chunk: {
          content: "text response",
          additional_kwargs: {
            reasoning: {
              summary: [{ index: 0 }],
            },
          },
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should return null when OpenAI summary text is empty", () => {
      const eventData = {
        chunk: {
          content: "text response",
          additional_kwargs: {
            reasoning: {
              summary: [{ text: "", index: 0 }],
            },
          },
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });
  });

  describe("Edge cases", () => {
    it("should return null when there is no reasoning content at all", () => {
      const eventData = {
        chunk: {
          content: [{ type: "text", text: "Hello" }],
          additional_kwargs: {},
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).toBeNull();
    });

    it("should prioritize Anthropic format over OpenAI", () => {
      const eventData = {
        chunk: {
          content: [{ thinking: "Anthropic thinking", type: "thinking", index: 0 }],
          additional_kwargs: {
            reasoning: {
              summary: [{ text: "OpenAI reasoning", index: 0 }],
            },
          },
        },
      };

      const result = resolveReasoningContent(eventData);

      expect(result).not.toBeNull();
      expect(result?.text).toBe("Anthropic thinking");
    });

    it("should throw when chunk is undefined", () => {
      const eventData = {};

      // The implementation throws when eventData.chunk is undefined
      // This documents the current behavior
      expect(() => resolveReasoningContent(eventData)).toThrow();
    });
  });
});

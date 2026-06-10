import { describe, it, expect, vi } from "vitest";
import { withCompression } from "../src/anthropic.js";

function mockFetch(body: unknown, status = 200): typeof globalThis.fetch {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  });
}

const COMPRESS_RESPONSE = {
  output: "compressed",
  output_tokens: 5,
  original_input_tokens: 20,
};

const SEARCH_RESPONSE = {
  results: [
    { url: "https://example.com", title: "Example", content: "Result content", score: 0.95 },
  ],
  query: "test query",
  search_time: 0.5,
  original_input_tokens: 100,
  output_tokens: 30,
};

function makeMockClient(createFn: Function) {
  return {
    messages: {
      create: createFn,
    },
  };
}

describe("withCompression webSearch", () => {
  it("webSearch false (default) does not modify tools", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const originalTools = [
      { type: "web_search_20250305", name: "web_search", max_uses: 3 },
      { name: "other_tool", input_schema: {} },
    ];
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [{ type: "text", text: "hello" }],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
      tools: originalTools,
    });

    const passedParams = createFn.mock.calls[0][0];
    // Should still have web_search_20250305
    expect(passedParams.tools.some((t: any) => t.type === "web_search_20250305")).toBe(true);
    // Should NOT have ttc_web_search
    expect(passedParams.tools.some((t: any) => t.name === "ttc_web_search")).toBe(false);
  });

  it("webSearch true removes any web_search_* version and adds ttc_web_search", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const originalTools = [
      { type: "web_search_20250305", name: "web_search", max_uses: 3 },
      { name: "other_tool", input_schema: {} },
    ];
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [{ type: "text", text: "hello" }],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
      tools: originalTools,
    });

    const passedParams = createFn.mock.calls[0][0];
    // Should NOT have any web_search_* type
    expect(passedParams.tools.some((t: any) => String(t.type ?? "").startsWith("web_search_"))).toBe(false);
    // Should have ttc_web_search
    expect(passedParams.tools.some((t: any) => t.name === "ttc_web_search")).toBe(true);
    // Should still have other_tool
    expect(passedParams.tools.some((t: any) => t.name === "other_tool")).toBe(true);
  });

  it("webSearch true removes web_search_20260209 variant", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [{ type: "text", text: "hello" }],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
      tools: [{ type: "web_search_20260209", name: "web_search" }],
    });

    const passedParams = createFn.mock.calls[0][0];
    expect(passedParams.tools.some((t: any) => String(t.type ?? "").startsWith("web_search_"))).toBe(false);
    expect(passedParams.tools.some((t: any) => t.name === "ttc_web_search")).toBe(true);
  });

  it("webSearch true adds ttc_web_search even when no tools provided", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [{ type: "text", text: "hello" }],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
    });

    const passedParams = createFn.mock.calls[0][0];
    expect(passedParams.tools).toHaveLength(1);
    expect(passedParams.tools[0].name).toBe("ttc_web_search");
  });

  it("webSearch true does not duplicate ttc_web_search if already present", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [{ type: "text", text: "hello" }],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
      tools: [{ name: "ttc_web_search", input_schema: {} }],
    });

    const passedParams = createFn.mock.calls[0][0];
    const searchTools = passedParams.tools.filter((t: any) => t.name === "ttc_web_search");
    expect(searchTools).toHaveLength(1);
  });

  it("tool loop handles search response and re-calls create", async () => {
    // The fetch mock handles both compress and search calls
    const fetchFn = vi.fn()
      .mockResolvedValueOnce({
        // First call: compress
        ok: true,
        status: 200,
        json: async () => COMPRESS_RESPONSE,
        text: async () => JSON.stringify(COMPRESS_RESPONSE),
      })
      .mockResolvedValueOnce({
        // Second call: search
        ok: true,
        status: 200,
        json: async () => SEARCH_RESPONSE,
        text: async () => JSON.stringify(SEARCH_RESPONSE),
      });

    let callCount = 0;
    const createFn = vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        // First call: model wants to use search
        return {
          stop_reason: "tool_use",
          content: [
            { type: "text", text: "Let me search for that." },
            {
              type: "tool_use",
              id: "toolu_123",
              name: "ttc_web_search",
              input: { query: "test query" },
            },
          ],
        };
      }
      // Second call: model produces final answer
      return {
        stop_reason: "end_turn",
        content: [{ type: "text", text: "Based on search results..." }],
      };
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "What is the latest news?" }],
    });

    // Should have called create twice (initial + after search)
    expect(createFn).toHaveBeenCalledTimes(2);

    // Second call should include the assistant message and tool results
    const secondCall = createFn.mock.calls[1][0];
    expect(secondCall.messages).toHaveLength(3); // original user + assistant + tool_result
    expect(secondCall.messages[1].role).toBe("assistant");
    expect(secondCall.messages[2].role).toBe("user");
    expect(secondCall.messages[2].content[0].type).toBe("tool_result");
    expect(secondCall.messages[2].content[0].tool_use_id).toBe("toolu_123");
    expect(secondCall.messages[2].content[0].content).toContain("Example");
    expect(secondCall.messages[2].content[0].content).toContain("https://example.com");

    // Final response should be the end_turn
    expect(response.stop_reason).toBe("end_turn");
  });

  it("tool loop handles multiple search calls in a single response", async () => {
    const fetchFn = vi.fn()
      .mockResolvedValueOnce({
        ok: true, status: 200,
        json: async () => COMPRESS_RESPONSE,
        text: async () => JSON.stringify(COMPRESS_RESPONSE),
      })
      .mockResolvedValueOnce({
        ok: true, status: 200,
        json: async () => SEARCH_RESPONSE,
        text: async () => JSON.stringify(SEARCH_RESPONSE),
      })
      .mockResolvedValueOnce({
        ok: true, status: 200,
        json: async () => ({
          ...SEARCH_RESPONSE,
          results: [{ url: "https://other.com", title: "Other", content: "Other result" }],
        }),
        text: async () => "{}",
      });

    let callCount = 0;
    const createFn = vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        return {
          stop_reason: "tool_use",
          content: [
            {
              type: "tool_use",
              id: "toolu_1",
              name: "ttc_web_search",
              input: { query: "query one" },
            },
            {
              type: "tool_use",
              id: "toolu_2",
              name: "ttc_web_search",
              input: { query: "query two" },
            },
          ],
        };
      }
      return {
        stop_reason: "end_turn",
        content: [{ type: "text", text: "Done." }],
      };
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "search" }],
    });

    expect(createFn).toHaveBeenCalledTimes(2);

    const secondCall = createFn.mock.calls[1][0];
    // Should have 2 tool results in the user message
    const toolResultMsg = secondCall.messages.find((m: any) => m.role === "user" && Array.isArray(m.content) && m.content[0]?.type === "tool_result");
    expect(toolResultMsg.content).toHaveLength(2);
    expect(toolResultMsg.content[0].tool_use_id).toBe("toolu_1");
    expect(toolResultMsg.content[1].tool_use_id).toBe("toolu_2");
  });

  it("does not loop when stop_reason is not tool_use", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [
        { type: "text", text: "No search needed." },
      ],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
    });

    expect(createFn).toHaveBeenCalledTimes(1);
    expect(response.stop_reason).toBe("end_turn");
  });

  it("does not loop for non-search tool_use", async () => {
    const fetchFn = mockFetch(COMPRESS_RESPONSE);
    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "tool_use",
      content: [
        {
          type: "tool_use",
          id: "toolu_other",
          name: "calculator",
          input: { expression: "2+2" },
        },
      ],
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "calculate 2+2" }],
    });

    // Should NOT loop since it's not a ttc_web_search tool
    expect(createFn).toHaveBeenCalledTimes(1);
    expect(response.stop_reason).toBe("tool_use");
  });
});

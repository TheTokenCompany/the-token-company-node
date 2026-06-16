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

  it("does not double-compress search results on multi-turn conversation", async () => {
    const compressCalls: string[] = [];
    const fetchFn = vi.fn().mockImplementation(async (_url: string, opts: any) => {
      // Decompress gzipped body to inspect the request
      let payload: any;
      try {
        const { gunzip } = await import("node:zlib");
        const { promisify } = await import("node:util");
        const gunzipAsync = promisify(gunzip);
        const buf = Buffer.from(opts.body);
        const decompressed = await gunzipAsync(buf);
        payload = JSON.parse(decompressed.toString());
      } catch {
        payload = JSON.parse(typeof opts.body === "string" ? opts.body : "{}");
      }

      const url = typeof _url === "string" ? _url : "";
      if (url.includes("/v1/search")) {
        return {
          ok: true, status: 200,
          json: async () => SEARCH_RESPONSE,
          text: async () => JSON.stringify(SEARCH_RESPONSE),
        };
      }
      // Compress endpoint — track what gets compressed
      if (payload.input) {
        compressCalls.push(payload.input);
      }
      return {
        ok: true, status: 200,
        json: async () => ({
          output: `[c]${payload.input ?? ""}`,
          output_tokens: 5,
          original_input_tokens: 20,
        }),
        text: async () => "{}",
      };
    });

    let turnCount = 0;
    const createFn = vi.fn().mockImplementation(async () => {
      turnCount++;
      if (turnCount === 1) {
        // Turn 1: model wants to search
        return {
          stop_reason: "tool_use",
          content: [
            { type: "tool_use", id: "toolu_s1", name: "ttc_web_search", input: { query: "latest news" } },
          ],
        };
      }
      if (turnCount === 2) {
        // Turn 1 continued: model answers after search
        return {
          stop_reason: "end_turn",
          content: [{ type: "text", text: "Here are the results." }],
        };
      }
      // Turn 2: model answers directly
      return {
        stop_reason: "end_turn",
        content: [{ type: "text", text: "Sure, more info." }],
      };
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    // Turn 1: triggers search
    const response1 = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "What is the latest news?" }],
    });
    expect(response1.stop_reason).toBe("end_turn");

    // Build turn 2 messages including the full conversation history
    // This simulates what a user would do: pass previous messages + new message
    const turn2Messages = [
      { role: "user", content: "What is the latest news?" },
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_s1", name: "ttc_web_search", input: { query: "latest news" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_s1", content: "Source: Example\nURL: https://example.com\nResult content" },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "Here are the results." }] },
      { role: "user", content: "Tell me more" },
    ];

    compressCalls.length = 0; // Reset tracking

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: turn2Messages,
    });

    // The search tool_result content should NOT appear in compression calls
    const searchContent = "Source: Example\nURL: https://example.com\nResult content";
    expect(compressCalls).not.toContain(searchContent);
    // But the user text messages should be compressed
    expect(compressCalls).toContain("What is the latest news?");
    expect(compressCalls).toContain("Tell me more");
  });

  it("does not skip compression when webSearch is false even with ttc_web_search tool_use in history", async () => {
    const compressCalls: string[] = [];
    const fetchFn = vi.fn().mockImplementation(async (_url: string, opts: any) => {
      let payload: any;
      try {
        const { gunzip } = await import("node:zlib");
        const { promisify } = await import("node:util");
        const gunzipAsync = promisify(gunzip);
        const buf = Buffer.from(opts.body);
        const decompressed = await gunzipAsync(buf);
        payload = JSON.parse(decompressed.toString());
      } catch {
        payload = JSON.parse(typeof opts.body === "string" ? opts.body : "{}");
      }
      if (payload.input) compressCalls.push(payload.input);
      return {
        ok: true, status: 200,
        json: async () => ({
          output: `[c]${payload.input ?? ""}`,
          output_tokens: 5,
          original_input_tokens: 20,
        }),
        text: async () => "{}",
      };
    });

    const createFn = vi.fn().mockResolvedValue({
      stop_reason: "end_turn",
      content: [{ type: "text", text: "ok" }],
    });

    // webSearch: false (default)
    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "toolu_s1", name: "ttc_web_search", input: { query: "q" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_s1", content: "search data" },
          ],
        },
      ],
    });

    // With webSearch off, the tool_result SHOULD be compressed (no skip logic)
    expect(compressCalls).toContain("search data");
  });

  it("multi-search preserves context across iterations", async () => {
    const fetchFn = vi.fn().mockImplementation(async (_url: string) => {
      const url = typeof _url === "string" ? _url : "";
      if (url.includes("/v1/search")) {
        return {
          ok: true, status: 200,
          json: async () => SEARCH_RESPONSE,
          text: async () => JSON.stringify(SEARCH_RESPONSE),
        };
      }
      return {
        ok: true, status: 200,
        json: async () => COMPRESS_RESPONSE,
        text: async () => JSON.stringify(COMPRESS_RESPONSE),
      };
    });

    let callCount = 0;
    const createFn = vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount <= 3) {
        return {
          stop_reason: "tool_use",
          content: [{
            type: "tool_use",
            id: `toolu_${callCount}`,
            name: "ttc_web_search",
            input: { query: `query ${callCount}` },
          }],
        };
      }
      return { stop_reason: "end_turn", content: [{ type: "text", text: "Done." }] };
    });

    const client = withCompression(makeMockClient(createFn), {
      compressionApiKey: "ttc-test",
      webSearch: true,
      fetch: fetchFn,
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Research AI" }],
    });

    expect(createFn).toHaveBeenCalledTimes(4);

    // The 4th call (final) must have all 3 assistant+tool_result pairs
    const lastCall = createFn.mock.calls[3][0];
    const assistantMsgs = lastCall.messages.filter((m: any) => m.role === "assistant");
    const toolResultMsgs = lastCall.messages.filter(
      (m: any) => m.role === "user" && Array.isArray(m.content) &&
        m.content.some((b: any) => b.type === "tool_result")
    );

    expect(assistantMsgs).toHaveLength(3);
    expect(toolResultMsgs).toHaveLength(3);
  });

  it("search compression stats are tracked", async () => {
    const fetchFn = vi.fn().mockImplementation(async (_url: string) => {
      const url = typeof _url === "string" ? _url : "";
      if (url.includes("/v1/search")) {
        return {
          ok: true, status: 200,
          json: async () => ({
            ...SEARCH_RESPONSE,
            original_input_tokens: 1000,
            output_tokens: 700,
          }),
          text: async () => "{}",
        };
      }
      // No-op compress (0 savings)
      return {
        ok: true, status: 200,
        json: async () => ({ output: "same", output_tokens: 10, original_input_tokens: 10 }),
        text: async () => "{}",
      };
    });

    let callCount = 0;
    const createFn = vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        return {
          stop_reason: "tool_use",
          content: [{
            type: "tool_use", id: "toolu_1", name: "ttc_web_search",
            input: { query: "test" },
          }],
        };
      }
      return { stop_reason: "end_turn", content: [{ type: "text", text: "Done." }] };
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

    // Search saved 300 tokens (1000→700), compress saved 0
    expect(client.compression.totalTokensSaved).toBeGreaterThanOrEqual(300);
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

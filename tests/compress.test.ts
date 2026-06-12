import { describe, it, expect, vi } from "vitest";
import { resolveAggressiveness, compressOpenAIMessages, compressAnthropicMessages, compressAISDKPrompt } from "../src/compress.js";
import { TheTokenCompany } from "../src/client.js";

function mockTTC(): TheTokenCompany {
  const ttc = { compress: vi.fn() } as unknown as TheTokenCompany;
  vi.mocked(ttc.compress).mockImplementation(async (text: string) => ({
    output: `[c]${text}`,
    outputTokens: 5,
    inputTokens: 20,
    tokensSaved: 15,
    compressionRatio: 4,
  }));
  return ttc;
}

describe("resolveAggressiveness", () => {
  it("expands float to default roles", () => {
    expect(resolveAggressiveness(0.3)).toEqual({ user: 0.3, system: 0.3, tool: 0.3 });
  });

  it("passes dict through", () => {
    const d = { user: 0.5 };
    expect(resolveAggressiveness(d)).toBe(d);
  });
});

describe("compressOpenAIMessages", () => {
  it("compresses user, skips assistant", async () => {
    const ttc = mockTTC();
    const messages = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
      { role: "user", content: "bye" },
    ];
    const result = await compressOpenAIMessages(ttc, messages, "bear-2", { user: 0.2 });
    expect(result[0].content).toBe("[c]hello");
    expect(result[1].content).toBe("hi there");
    expect(result[2].content).toBe("[c]bye");
  });

  it("compresses tool and function roles with tool aggressiveness", async () => {
    const ttc = mockTTC();
    const messages = [
      { role: "tool", content: "tool output" },
      { role: "function", content: "fn output" },
    ];
    const result = await compressOpenAIMessages(ttc, messages, "bear-2", { tool: 0.7 });
    expect(result[0].content).toBe("[c]tool output");
    expect(result[1].content).toBe("[c]fn output");
  });

  it("skips roles not in dict", async () => {
    const ttc = mockTTC();
    const messages = [
      { role: "system", content: "sys prompt" },
      { role: "user", content: "hello" },
    ];
    const result = await compressOpenAIMessages(ttc, messages, "bear-2", { user: 0.2 });
    expect(result[0].content).toBe("sys prompt");
    expect(result[1].content).toBe("[c]hello");
  });

  it("compresses text blocks in multimodal content", async () => {
    const ttc = mockTTC();
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "describe this" },
          { type: "image_url", image_url: { url: "https://..." } },
        ],
      },
    ];
    const result = await compressOpenAIMessages(ttc, messages, "bear-2", { user: 0.2 });
    const blocks = result[0].content as any[];
    expect(blocks[0].text).toBe("[c]describe this");
    expect(blocks[1].type).toBe("image_url");
  });
});

describe("compressAnthropicMessages", () => {
  it("compresses user text, skips assistant", async () => {
    const ttc = mockTTC();
    const messages = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { user: 0.2 });
    expect(result[0].content).toBe("[c]hello");
    expect(result[1].content).toBe("hi there");
  });

  it("compresses tool_result blocks", async () => {
    const ttc = mockTTC();
    const messages = [
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: "search results..." },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { tool: 0.3 });
    const block = (result[0].content as any[])[0];
    expect(block.content).toBe("[c]search results...");
    expect(block.tool_use_id).toBe("t1");
  });

  it("uses different aggressiveness for text vs tool_result", async () => {
    const ttc = mockTTC();
    const calls: number[] = [];
    vi.mocked(ttc.compress).mockImplementation(async (text: string, opts: any) => {
      calls.push(opts.aggressiveness);
      return { output: text, outputTokens: 5, inputTokens: 20, tokensSaved: 15, compressionRatio: 4 };
    });

    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "hello" },
          { type: "tool_result", tool_use_id: "t1", content: "data" },
        ],
      },
    ];
    await compressAnthropicMessages(ttc, messages, "bear-2", { user: 0.1, tool: 0.8 });
    expect(calls).toEqual([0.1, 0.8]);
  });

  it("skips tool_result when tool not in dict", async () => {
    const ttc = mockTTC();
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "hello" },
          { type: "tool_result", tool_use_id: "t1", content: "raw data" },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { user: 0.2 });
    const blocks = result[0].content as any[];
    expect(blocks[0].text).toBe("[c]hello");
    expect(blocks[1].content).toBe("raw data");
  });
});

describe("compressAnthropicMessages skipToolName", () => {
  it("skips tool_result whose tool_use_id matches a ttc_web_search tool_use in assistant messages", async () => {
    const ttc = mockTTC();
    const messages = [
      { role: "user", content: "search for news" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me search." },
          { type: "tool_use", id: "toolu_search1", name: "ttc_web_search", input: { query: "news" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_search1", content: "Already compressed search content" },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { user: 0.2, tool: 0.3 }, { skipToolName: "ttc_web_search" });
    // User text message should be compressed
    expect(result[0].content).toBe("[c]search for news");
    // tool_result for search should NOT be compressed
    const toolBlock = (result[2].content as any[])[0];
    expect(toolBlock.content).toBe("Already compressed search content");
  });

  it("still compresses tool_results from non-search tools", async () => {
    const ttc = mockTTC();
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_search1", name: "ttc_web_search", input: { query: "q" } },
          { type: "tool_use", id: "toolu_calc1", name: "calculator", input: { expr: "2+2" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_search1", content: "search data" },
          { type: "tool_result", tool_use_id: "toolu_calc1", content: "calculator data" },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { tool: 0.3 }, { skipToolName: "ttc_web_search" });
    const blocks = result[1].content as any[];
    // Search result skipped
    expect(blocks[0].content).toBe("search data");
    // Calculator result compressed
    expect(blocks[1].content).toBe("[c]calculator data");
  });

  it("skips multiple search tool_results across multiple turns", async () => {
    const ttc = mockTTC();
    const messages = [
      // Turn 1 search
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_s1", name: "ttc_web_search", input: { query: "q1" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_s1", content: "result 1" },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "Let me search more." }] },
      // Turn 2 search
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_s2", name: "ttc_web_search", input: { query: "q2" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_s2", content: "result 2" },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { tool: 0.3 }, { skipToolName: "ttc_web_search" });
    expect((result[1].content as any[])[0].content).toBe("result 1");
    expect((result[4].content as any[])[0].content).toBe("result 2");
    expect(ttc.compress).not.toHaveBeenCalledWith("result 1", expect.anything());
    expect(ttc.compress).not.toHaveBeenCalledWith("result 2", expect.anything());
  });

  it("compresses everything normally when skipToolName is not set", async () => {
    const ttc = mockTTC();
    const messages = [
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
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { tool: 0.3 });
    // Without skipToolName, search results get compressed
    expect((result[1].content as any[])[0].content).toBe("[c]search data");
  });

  it("works with JSON-serialized messages (simulates server switch / client recreation)", async () => {
    const ttc = mockTTC();
    const originalMessages = [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_abc", name: "ttc_web_search", input: { query: "test" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_abc", content: "pre-compressed result" },
        ],
      },
      { role: "user", content: "follow up question" },
    ];
    // Simulate serialization round-trip (what happens when switching servers)
    const messages = JSON.parse(JSON.stringify(originalMessages));
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { user: 0.2, tool: 0.3 }, { skipToolName: "ttc_web_search" });
    // Search tool_result should still be skipped after serialization
    expect((result[1].content as any[])[0].content).toBe("pre-compressed result");
    // Regular user text should still be compressed
    expect(result[2].content).toBe("[c]follow up question");
  });

  it("handles tool_result with array content blocks", async () => {
    const ttc = mockTTC();
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_s1", name: "ttc_web_search", input: { query: "q" } },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "toolu_s1",
            content: [{ type: "text", text: "search result text" }],
          },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { tool: 0.3 }, { skipToolName: "ttc_web_search" });
    const inner = (result[1].content as any[])[0].content;
    // The entire tool_result block should be returned as-is
    expect(inner[0].text).toBe("search result text");
  });

  it("does not skip tool_result with no matching tool_use in assistant messages", async () => {
    const ttc = mockTTC();
    const messages = [
      // No assistant message with tool_use — orphaned tool_result
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_orphan", content: "some data" },
        ],
      },
    ];
    const result = await compressAnthropicMessages(ttc, messages, "bear-2", { tool: 0.3 }, { skipToolName: "ttc_web_search" });
    // Should be compressed since there's no matching search tool_use
    expect((result[0].content as any[])[0].content).toBe("[c]some data");
  });
});

describe("compressAISDKPrompt", () => {
  it("compresses system and user, skips assistant", async () => {
    const ttc = mockTTC();
    const prompt = [
      { role: "system" as const, content: "be helpful" },
      { role: "user" as const, content: "hello" },
      { role: "assistant" as const, content: "hi" },
    ];
    const result = await compressAISDKPrompt(ttc, prompt, "bear-2", { system: 0.1, user: 0.2 });
    expect(result[0].content).toBe("[c]be helpful");
    expect(result[1].content).toBe("[c]hello");
    expect(result[2].content).toBe("hi");
  });

  it("compresses text parts in user messages", async () => {
    const ttc = mockTTC();
    const prompt = [
      {
        role: "user" as const,
        content: [
          { type: "text" as const, text: "describe this" },
          { type: "image" as const, image: "data:..." },
        ],
      },
    ];
    const result = await compressAISDKPrompt(ttc, prompt, "bear-2", { user: 0.2 });
    const parts = result[0].content as any[];
    expect(parts[0].text).toBe("[c]describe this");
    expect(parts[1].type).toBe("image");
  });
});

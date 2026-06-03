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

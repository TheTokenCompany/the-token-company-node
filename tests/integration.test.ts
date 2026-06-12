import { describe, it, expect, beforeAll } from "vitest";
import { TheTokenCompany } from "../src/client.js";
import { withCompression } from "../src/anthropic.js";

const TTC_API_KEY = process.env.TTC_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const skip = !TTC_API_KEY || !ANTHROPIC_API_KEY;

let Anthropic: any;

describe.skipIf(skip)("integration: real endpoints", () => {
  let ttc: TheTokenCompany;

  beforeAll(async () => {
    ttc = new TheTokenCompany({ apiKey: TTC_API_KEY! });
    const mod = await import("@anthropic-ai/sdk");
    Anthropic = mod.default;
  });

  it("TTC compress endpoint works", async () => {
    const result = await ttc.compress(
      "The quick brown fox jumps over the lazy dog. This is a test of the compression system to verify it correctly reduces token count.",
      { model: "bear-2", aggressiveness: 0.3 }
    );
    expect(result.output).toBeTruthy();
    expect(result.outputTokens).toBeLessThanOrEqual(result.inputTokens);
    expect(result.tokensSaved).toBeGreaterThanOrEqual(0);
  });

  it("TTC search endpoint returns compressed results", async () => {
    const result = await ttc.search("What is TypeScript?", {
      maxResults: 3,
      model: "bear-2",
      aggressiveness: 0.3,
    });
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].url).toBeTruthy();
    expect(result.results[0].title).toBeTruthy();
    expect(result.results[0].content).toBeTruthy();
    expect(result.tokensSaved).toBeGreaterThanOrEqual(0);
  });

  it("withCompression wrapper compresses messages and calls Anthropic", async () => {
    const anthropic = new Anthropic();
    const client = withCompression(anthropic, {
      compressionApiKey: TTC_API_KEY!,
    });

    const response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 50,
      messages: [{ role: "user", content: "Say hello in one word." }],
    });

    expect(response.stop_reason).toBe("end_turn");
    expect(response.content[0]).toHaveProperty("text");
    expect(client.compression.totalTokensSaved).toBeGreaterThanOrEqual(0);
  });

  it("webSearch: model can search and return results", async () => {
    const anthropic = new Anthropic();
    const client = withCompression(anthropic, {
      compressionApiKey: TTC_API_KEY!,
      webSearch: true,
    });

    const response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 300,
      system: "You have a web search tool. Always use it to answer questions. Be very brief.",
      messages: [{ role: "user", content: "What is the current population of Finland? Search for it." }],
    });

    expect(response.stop_reason).toBe("end_turn");
    const text = response.content.find((b: any) => b.type === "text");
    expect(text).toBeTruthy();
  }, 30_000);

  it("search results are NOT double-compressed in multi-turn", async () => {
    const compressedTexts: string[] = [];
    const originalFetch = globalThis.fetch;
    const interceptFetch: typeof globalThis.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : (input as Request).url;
      const response = await originalFetch(input, init);

      if (url.includes("/v1/compress")) {
        const cloned = response.clone();
        try {
          const json = await cloned.json() as { output?: string };
          if (json.output) compressedTexts.push(json.output);
        } catch {}
      }
      return response;
    };

    const anthropic = new Anthropic();
    const client = withCompression(anthropic, {
      compressionApiKey: TTC_API_KEY!,
      webSearch: true,
      fetch: interceptFetch,
    });

    const turn1Response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 300,
      system: "You have a web search tool. Always use it first. Be very brief in your answers.",
      messages: [{ role: "user", content: "What is the capital of Iceland? Search for it." }],
    });
    expect(turn1Response.stop_reason).toBe("end_turn");

    const searchToolUseId = "toolu_turn1";
    const searchResultContent = "Source: Wikipedia\nURL: https://en.wikipedia.org/wiki/Reykjavik\nReykjavik is the capital of Iceland.";

    const turn2Messages: any[] = [
      { role: "user", content: "What is the capital of Iceland? Search for it." },
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: searchToolUseId, name: "ttc_web_search", input: { query: "capital of Iceland" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: searchToolUseId, content: searchResultContent },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "The capital of Iceland is Reykjavik." }],
      },
      { role: "user", content: "And what is the population of that city?" },
    ];

    compressedTexts.length = 0;

    const turn2Response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 300,
      system: "You have a web search tool. You can use it if needed. Be very brief.",
      messages: turn2Messages,
    });

    expect(turn2Response.stop_reason).toBe("end_turn");

    const searchContentWasCompressed = compressedTexts.some(
      (t) => t.includes("Reykjavik is the capital of Iceland")
    );
    expect(searchContentWasCompressed).toBe(false);

    expect(compressedTexts.length).toBeGreaterThan(0);
  }, 30_000);

  it("regular tool_results ARE still compressed in multi-turn", async () => {
    const compressInputs: string[] = [];
    const originalFetch = globalThis.fetch;
    const interceptFetch: typeof globalThis.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : (input as Request).url;

      if (url.includes("/v1/compress") && init?.body) {
        try {
          const { gunzip } = await import("node:zlib");
          const { promisify } = await import("node:util");
          const gunzipAsync = promisify(gunzip);
          const buf = Buffer.from(init.body as ArrayBuffer);
          const decompressed = await gunzipAsync(buf);
          const payload = JSON.parse(decompressed.toString());
          if (payload.input) compressInputs.push(payload.input);
        } catch {}
      }
      return originalFetch(input, init);
    };

    const anthropic = new Anthropic();
    const client = withCompression(anthropic, {
      compressionApiKey: TTC_API_KEY!,
      webSearch: true,
      fetch: interceptFetch,
    });

    const messages: any[] = [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_calc", name: "calculator", input: { expr: "2+2" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_calc", content: "The calculator returned the result: 4" },
        ],
      },
      { role: "user", content: "Thanks, what about 3+3?" },
    ];

    const response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 100,
      messages,
    });

    expect(response.stop_reason).toBe("end_turn");

    const calcWasCompressed = compressInputs.some(
      (t) => t.includes("calculator returned the result")
    );
    expect(calcWasCompressed).toBe(true);
  }, 30_000);
});

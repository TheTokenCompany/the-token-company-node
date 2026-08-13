import { describe, it, expect, vi } from "vitest";
import { withCompression as withAnthropic } from "../src/anthropic.js";
import { withCompression as withOpenAI } from "../src/openai.js";
import { compressionMiddleware } from "../src/ai-sdk.js";

/**
 * These tests assert the ACTUAL outgoing request the wrappers send to
 * /v1/chat/compress — the boundary that decides whether the server compresses
 * agent/assistant turns. The default (a scalar aggressiveness) must now expand
 * to include the "assistant" role; a per-role dict without "assistant" must
 * omit it (the KV-cache opt-out).
 */

function chatCompressResponse(messages: unknown[]) {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      messages,
      system: undefined,
      original_input_tokens: 20,
      output_tokens: 10,
      cache_hits: 0,
      cache_misses: Array.isArray(messages) ? messages.length : 1,
      compression_time: 0.01,
    }),
    text: async () => "",
  };
}

/** Capture and decode the gzipped body sent to /v1/chat/compress. */
function capturingFetch(): { fetch: typeof globalThis.fetch; body: () => Promise<any> } {
  const calls: any[] = [];
  const fetch = vi.fn(async (_url: string, opts: any) => {
    calls.push(opts);
    // Server echoes messages back so the wrapper's downstream call is well-formed.
    const decoded = await decode(opts);
    return chatCompressResponse(decoded.messages ?? []) as any;
  }) as unknown as typeof globalThis.fetch;
  return { fetch, body: async () => decode(calls[0]) };
}

async function decode(opts: any): Promise<any> {
  const { gunzip } = await import("node:zlib");
  const { promisify } = await import("node:util");
  const gunzipAsync = promisify(gunzip);
  try {
    return JSON.parse((await gunzipAsync(Buffer.from(opts.body))).toString());
  } catch {
    return JSON.parse(typeof opts?.body === "string" ? opts.body : "{}");
  }
}

const MESSAGES = [
  { role: "user", content: "hi" },
  { role: "assistant", content: "a long agent response" },
  { role: "user", content: "follow up" },
];

describe("assistant compressed by default — outgoing payload", () => {
  it("anthropic wrapper sends assistant in aggressiveness by default", async () => {
    const cap = capturingFetch();
    const client: any = { messages: { create: vi.fn().mockResolvedValue({ content: [] }) } };
    const wrapped = withAnthropic(client, { compressionApiKey: "ttc-test", fetch: cap.fetch });
    await wrapped.messages.create({ model: "claude-sonnet-4-6", max_tokens: 16, messages: MESSAGES });

    const body = await cap.body();
    expect(body.aggressiveness).toEqual({ user: 0.2, system: 0.2, tool: 0.2, assistant: 0.2 });
  });

  it("openai wrapper sends assistant in aggressiveness by default", async () => {
    const cap = capturingFetch();
    const client: any = { chat: { completions: { create: vi.fn().mockResolvedValue({}) } } };
    const wrapped = withOpenAI(client, { compressionApiKey: "ttc-test", fetch: cap.fetch });
    await wrapped.chat.completions.create({ model: "gpt-4o", messages: MESSAGES });

    const body = await cap.body();
    expect(body.aggressiveness.assistant).toBe(0.2);
  });

  it("ai-sdk middleware sends assistant in aggressiveness by default", async () => {
    const cap = capturingFetch();
    const mw = compressionMiddleware({ compressionApiKey: "ttc-test", fetch: cap.fetch });
    await mw.transformParams!({
      type: "generate",
      params: { prompt: MESSAGES } as any,
      model: {} as any,
    });

    const body = await cap.body();
    expect(body.aggressiveness.assistant).toBe(0.2);
  });

  it("per-role dict without assistant omits it (KV-cache opt-out)", async () => {
    const cap = capturingFetch();
    const client: any = { chat: { completions: { create: vi.fn().mockResolvedValue({}) } } };
    const wrapped = withOpenAI(client, {
      compressionApiKey: "ttc-test",
      fetch: cap.fetch,
      aggressiveness: { user: 0.2, system: 0.2, tool: 0.2 },
    });
    await wrapped.chat.completions.create({ model: "gpt-4o", messages: MESSAGES });

    const body = await cap.body();
    expect("assistant" in body.aggressiveness).toBe(false);
  });
});

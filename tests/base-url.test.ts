import { describe, it, expect, vi } from "vitest";
import { withCompression as withAnthropic } from "../src/anthropic.js";
import { withCompression as withOpenAI } from "../src/openai.js";
import { compressionMiddleware } from "../src/ai-sdk.js";

/** Every wrapper must send /v1/chat/compress to the `baseUrl` it was given. */

const BASE_URL = "https://ttc.internal.example";
const MESSAGES = [{ role: "user", content: "hi" }];

function urlRecordingFetch() {
  return vi.fn(async (_url: string) => ({
    ok: true,
    status: 200,
    json: async () => ({
      messages: MESSAGES,
      system: null,
      original_input_tokens: 2,
      output_tokens: 1,
      cache_hits: 0,
      cache_misses: 1,
      compression_time: 0,
    }),
    text: async () => "",
  }));
}

describe("baseUrl option", () => {
  it("anthropic wrapper uses baseUrl", async () => {
    const fetch = urlRecordingFetch();
    const client: any = { messages: { create: vi.fn().mockResolvedValue({ content: [] }) } };
    const wrapped = withAnthropic(client, { compressionApiKey: "ttc-test", baseUrl: BASE_URL, fetch: fetch as any });
    await wrapped.messages.create({ model: "m", max_tokens: 16, messages: MESSAGES });
    expect(fetch.mock.calls[0][0]).toBe(`${BASE_URL}/v1/chat/compress`);
  });

  it("openai wrapper uses baseUrl", async () => {
    const fetch = urlRecordingFetch();
    const client: any = { chat: { completions: { create: vi.fn().mockResolvedValue({}) } } };
    const wrapped = withOpenAI(client, { compressionApiKey: "ttc-test", baseUrl: BASE_URL, fetch: fetch as any });
    await wrapped.chat.completions.create({ model: "m", messages: MESSAGES });
    expect(fetch.mock.calls[0][0]).toBe(`${BASE_URL}/v1/chat/compress`);
  });

  it("ai-sdk middleware uses baseUrl", async () => {
    const fetch = urlRecordingFetch();
    const mw = compressionMiddleware({ compressionApiKey: "ttc-test", baseUrl: BASE_URL, fetch: fetch as any });
    await mw.transformParams!({ type: "generate", params: { prompt: MESSAGES } as any, model: {} as any });
    expect(fetch.mock.calls[0][0]).toBe(`${BASE_URL}/v1/chat/compress`);
  });
});

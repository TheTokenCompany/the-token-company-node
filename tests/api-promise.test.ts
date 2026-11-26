import { describe, it, expect, vi } from "vitest";
import { withCompression as withAnthropic } from "../src/anthropic.js";
import { withCompression as withOpenAI } from "../src/openai.js";

/**
 * The SDKs' create() returns an APIPromise, and their own helpers call methods
 * on it: messages.stream() calls create(...).withResponse(), and OpenAI's
 * chat.completions.parse() calls create(...)._thenUnwrap(). The wrapped create()
 * must keep those working, and still send the compressed messages.
 */

const RESPONSE = { status: 200, headers: new Map([["request-id", "req_1"]]) };

/** Minimal stand-in for the SDKs' APIPromise. */
function fakeAPIPromise(data: any): any {
  return Object.assign(Promise.resolve(data), {
    withResponse: async () => ({ data, response: RESPONSE }),
    asResponse: async () => RESPONSE,
    _thenUnwrap: (fn: (d: any) => any) => fakeAPIPromise(fn(data)),
  });
}

/** /v1/chat/compress mock that marks every message as compressed. */
const compressFetch = vi.fn(async (_url: string, opts: any) => {
  const { gunzip } = await import("node:zlib");
  const { promisify } = await import("node:util");
  const body = JSON.parse((await promisify(gunzip)(Buffer.from(opts.body))).toString());
  return {
    ok: true,
    status: 200,
    json: async () => ({
      messages: body.messages.map((m: any) => ({ ...m, content: "compressed" })),
      system: null,
      original_input_tokens: 20,
      output_tokens: 5,
      cache_hits: 0,
      cache_misses: body.messages.length,
      compression_time: 0,
    }),
    text: async () => "",
  };
}) as unknown as typeof globalThis.fetch;

const OPTIONS = { compressionApiKey: "ttc-test", fetch: compressFetch };
const MESSAGES = [{ role: "user", content: "a long prompt" }];

describe("anthropic wrapper keeps the APIPromise shape", () => {
  function makeClient() {
    const create = vi.fn((params: any) => fakeAPIPromise({ content: [], stream: !!params.stream }));
    return { create, client: withAnthropic({ messages: { create } } as any, OPTIONS) };
  }

  it("create(...).withResponse() works, as messages.stream() needs", async () => {
    const { create, client } = makeClient();
    const { data, response } = await client.messages
      .create({ model: "m", max_tokens: 16, stream: true, messages: MESSAGES })
      .withResponse();
    expect(data.stream).toBe(true);
    expect(response).toBe(RESPONSE);
    expect(create.mock.calls[0][0].messages[0].content).toBe("compressed");
  });

  it("create(...).asResponse() works", async () => {
    const { client } = makeClient();
    const response = await client.messages
      .create({ model: "m", max_tokens: 16, messages: MESSAGES })
      .asResponse();
    expect(response).toBe(RESPONSE);
  });

  it("awaiting create(...) still returns the data", async () => {
    const { client } = makeClient();
    const data = await client.messages.create({ model: "m", max_tokens: 16, messages: MESSAGES });
    expect(data).toEqual({ content: [], stream: false });
    expect(client.compression.totalTokensSaved).toBe(15);
  });
});

describe("anthropic wrapper with webSearch", () => {
  it("create(...).withResponse() returns the answer after the search, not the tool call", async () => {
    const toolCall = {
      stop_reason: "tool_use",
      content: [{ type: "tool_use", id: "tu_1", name: "ttc_web_search", input: { query: "q" } }],
    };
    const answer = { stop_reason: "end_turn", content: [{ type: "text", text: "answer" }] };
    const create = vi.fn().mockReturnValueOnce(fakeAPIPromise(toolCall)).mockReturnValueOnce(fakeAPIPromise(answer));
    const fetch = vi.fn(async (url: string, opts: any) => {
      if (url.includes("/v1/search")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ results: [], query: "q", search_time: 0, original_input_tokens: 0, output_tokens: 0 }),
          text: async () => "",
        };
      }
      return (compressFetch as any)(url, opts);
    }) as unknown as typeof globalThis.fetch;
    const client = withAnthropic({ messages: { create } } as any, { ...OPTIONS, fetch, webSearch: true });

    const { data, response } = await client.messages
      .create({ model: "m", max_tokens: 16, messages: MESSAGES })
      .withResponse();

    expect(data).toEqual(answer);
    expect(response).toBe(RESPONSE);
    expect(create).toHaveBeenCalledTimes(2);
  });
});

describe("openai wrapper keeps the APIPromise shape", () => {
  function makeClient() {
    const create = vi.fn((params: any) => fakeAPIPromise({ choices: [], stream: !!params.stream }));
    return { create, client: withOpenAI({ chat: { completions: { create } } } as any, OPTIONS) };
  }

  it("create({ stream: true }).withResponse() works", async () => {
    const { create, client } = makeClient();
    const { data, response } = await client.chat.completions
      .create({ model: "m", stream: true, messages: MESSAGES })
      .withResponse();
    expect(data.stream).toBe(true);
    expect(response).toBe(RESPONSE);
    expect(create.mock.calls[0][0].messages[0].content).toBe("compressed");
  });

  it("create(...).asResponse() works", async () => {
    const { client } = makeClient();
    expect(await client.chat.completions.create({ model: "m", messages: MESSAGES }).asResponse()).toBe(RESPONSE);
  });

  it("create(...)._thenUnwrap() works, as chat.completions.parse() needs", async () => {
    const { client } = makeClient();
    const parsed = (client.chat.completions.create({ model: "m", messages: MESSAGES }) as any)
      ._thenUnwrap((c: any) => ({ ...c, parsed: true }));
    expect(await parsed).toEqual({ choices: [], stream: false, parsed: true });
    expect((await parsed.withResponse()).data.parsed).toBe(true);
  });
});

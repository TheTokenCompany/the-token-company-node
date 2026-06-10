import { describe, it, expect, vi } from "vitest";
import { TheTokenCompany } from "../src/client.js";
import { AuthenticationError, APIError } from "../src/errors.js";

function mockFetch(body: unknown, status = 200): typeof globalThis.fetch {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  });
}

const SEARCH_RESPONSE = {
  results: [
    { url: "https://example.com", title: "Example", content: "Compressed result", score: 0.95 },
    { url: "https://other.com", title: "Other", content: "Another result" },
  ],
  query: "test query",
  search_time: 1.23,
  original_input_tokens: 500,
  output_tokens: 120,
};

describe("search", () => {
  it("sends correct request and maps response", async () => {
    const fetchFn = mockFetch(SEARCH_RESPONSE);
    const ttc = new TheTokenCompany({
      apiKey: "ttc-test",
      gzip: false,
      fetch: fetchFn,
    });

    const result = await ttc.search("test query");

    expect(fetchFn).toHaveBeenCalledOnce();
    const [url, init] = vi.mocked(fetchFn).mock.calls[0];
    expect(url).toBe("https://api.thetokencompany.com/v1/search");
    expect(init?.method).toBe("POST");

    const body = JSON.parse(init?.body as string);
    expect(body.query).toBe("test query");
    expect(body.max_results).toBe(5);
    expect(body.search_depth).toBe("basic");
    expect(body.include_raw_content).toBe(false);
    expect(body.model).toBe("bear-2");
    expect(body.compression_settings.aggressiveness).toBe(0.3);

    expect(result.results).toHaveLength(2);
    expect(result.results[0].url).toBe("https://example.com");
    expect(result.results[0].score).toBe(0.95);
    expect(result.results[1].score).toBeUndefined();
    expect(result.query).toBe("test query");
    expect(result.searchTime).toBe(1.23);
    expect(result.originalInputTokens).toBe(500);
    expect(result.outputTokens).toBe(120);
    expect(result.tokensSaved).toBe(380);
  });

  it("uses custom baseUrl", async () => {
    const fetchFn = mockFetch(SEARCH_RESPONSE);
    const ttc = new TheTokenCompany({
      apiKey: "ttc-test",
      baseUrl: "https://custom.api.example.com",
      gzip: false,
      fetch: fetchFn,
    });

    await ttc.search("test query");

    const [url] = vi.mocked(fetchFn).mock.calls[0];
    expect(url).toBe("https://custom.api.example.com/v1/search");
  });

  it("passes custom options through", async () => {
    const fetchFn = mockFetch(SEARCH_RESPONSE);
    const ttc = new TheTokenCompany({
      apiKey: "ttc-test",
      gzip: false,
      fetch: fetchFn,
    });

    await ttc.search("test query", {
      maxResults: 10,
      searchDepth: "advanced",
      includeRawContent: true,
      model: "bear-1.2",
      aggressiveness: 0.7,
      appId: "my-app",
    });

    const body = JSON.parse(vi.mocked(fetchFn).mock.calls[0][1]?.body as string);
    expect(body.max_results).toBe(10);
    expect(body.search_depth).toBe("advanced");
    expect(body.include_raw_content).toBe(true);
    expect(body.model).toBe("bear-1.2");
    expect(body.compression_settings.aggressiveness).toBe(0.7);
    expect(body.app_id).toBe("my-app");
  });

  it("uses constructor appId when not overridden", async () => {
    const fetchFn = mockFetch(SEARCH_RESPONSE);
    const ttc = new TheTokenCompany({
      apiKey: "ttc-test",
      gzip: false,
      appId: "default-app",
      fetch: fetchFn,
    });

    await ttc.search("test query");

    const body = JSON.parse(vi.mocked(fetchFn).mock.calls[0][1]?.body as string);
    expect(body.app_id).toBe("default-app");
  });

  it("throws AuthenticationError on 401", async () => {
    const fetchFn = mockFetch({ detail: "Invalid API key" }, 401);
    const ttc = new TheTokenCompany({
      apiKey: "bad-key",
      gzip: false,
      fetch: fetchFn,
    });

    await expect(ttc.search("test query")).rejects.toThrow(AuthenticationError);
  });

  it("throws APIError on 500", async () => {
    const fetchFn = mockFetch({ detail: "Internal server error" }, 500);
    const ttc = new TheTokenCompany({
      apiKey: "ttc-test",
      gzip: false,
      fetch: fetchFn,
    });

    await expect(ttc.search("test query")).rejects.toThrow(APIError);
  });
});

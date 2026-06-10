import { TheTokenCompany } from "./client.js";
import { StatsTTC, compressAnthropicMessages, resolveAggressiveness } from "./compress.js";
import type { WithCompressionOptions, SearchResultItem } from "./types.js";
import { CompressionStats } from "./types.js";

const TTC_SEARCH_TOOL = {
  name: "ttc_web_search",
  description: "Search the web for current information. Use this when you need up-to-date facts, prices, news, or any information that may have changed after your training cutoff.",
  input_schema: {
    type: "object" as const,
    properties: {
      query: {
        type: "string",
        description: "The search query"
      }
    },
    required: ["query"]
  }
};

function injectSearchTool(params: any): any {
  let tools = params.tools ? [...params.tools] : [];
  tools = tools.filter((t: any) => t.type !== "web_search_20250305");
  if (!tools.some((t: any) => t.name === "ttc_web_search")) {
    tools.push(TTC_SEARCH_TOOL);
  }
  return { ...params, tools };
}

function hasSearchToolUse(response: any): boolean {
  if (response.stop_reason !== "tool_use") return false;
  return response.content?.some(
    (b: any) => b.type === "tool_use" && b.name === "ttc_web_search"
  );
}

function formatSearchResults(results: SearchResultItem[]): string {
  return results.map(r =>
    `Source: ${r.title}\nURL: ${r.url}\n${r.content}`
  ).join("\n\n");
}

async function handleSearchLoop(
  response: any, params: any, originalCreate: Function,
  ttcClient: TheTokenCompany, rest: any[]
): Promise<any> {
  while (hasSearchToolUse(response)) {
    const messages = [...(params.messages || [])];

    // Add assistant response
    const assistantContent = response.content.map((b: any) => {
      if (typeof b.toJSON === 'function') return b.toJSON();
      return b;
    });
    messages.push({ role: "assistant", content: assistantContent });

    // Build tool results
    const toolResults: any[] = [];
    for (const block of response.content) {
      if (block.type === "tool_use" && block.name === "ttc_web_search") {
        const query = block.input?.query || "";
        const searchResult = await ttcClient.search(query);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: formatSearchResults(searchResult.results),
        });
      }
    }

    if (toolResults.length > 0) {
      messages.push({ role: "user", content: toolResults });
    }

    const newParams = { ...params, messages };
    response = await originalCreate(newParams, ...rest);
  }

  return response;
}

/**
 * Wrap an Anthropic client to auto-compress non-assistant messages.
 *
 * Compresses the `system` parameter and all non-assistant messages.
 *
 * ```ts
 * import Anthropic from "@anthropic-ai/sdk";
 * import { withCompression } from "the-token-company/anthropic";
 *
 * const client = withCompression(new Anthropic(), { compressionApiKey: "ttc-..." });
 * const response = await client.messages.create({
 *   model: "claude-sonnet-4-6",
 *   max_tokens: 1024,
 *   system: "You are a helpful assistant...",
 *   messages: [{ role: "user", content: "..." }],
 * });
 * console.log(client.compression.totalTokensSaved);
 * ```
 *
 * Assistant messages pass through unchanged to preserve the provider's KV cache.
 */
export function withCompression<T extends { messages: { create: Function } }>(
  client: T,
  options: WithCompressionOptions
): T & { compression: CompressionStats } {
  const stats = new CompressionStats();
  const ttcClient = new TheTokenCompany({ apiKey: options.compressionApiKey, baseUrl: options.baseUrl, appId: options.appId, fetch: options.fetch });
  const compressor = new StatsTTC(ttcClient, stats);
  const model = options.model ?? "bear-2";
  const roleAggr = resolveAggressiveness(options.aggressiveness ?? 0.2);
  if (options.compressAssistant && !("assistant" in roleAggr)) {
    roleAggr["assistant"] = roleAggr["user"] ?? 0.2;
  }
  const systemAggr = roleAggr["system"];
  const stripServerToolResults = options.stripServerToolResults ?? false;
  const webSearch = options.webSearch ?? false;
  const originalCreate = client.messages.create.bind(client.messages);

  client.messages.create = async function (params: any, ...rest: any[]) {
    stats._startTurn();
    if (params?.messages) {
      params = {
        ...params,
        messages: await compressAnthropicMessages(compressor, params.messages, model, roleAggr, { stripServerToolResults }),
      };
    }
    if (systemAggr != null && typeof params?.system === "string" && params.system.trim()) {
      const result = await compressor.compress(params.system, { model, aggressiveness: systemAggr });
      params = { ...params, system: result.output };
    }

    // Inject search tool
    if (webSearch) {
      params = injectSearchTool(params);
    }

    stats._endTurn();
    let response = await originalCreate(params, ...rest);

    // Handle search tool loop
    if (webSearch) {
      response = await handleSearchLoop(response, params, originalCreate, ttcClient, rest);
    }

    return response;
  } as any;

  (client as any).compression = stats;
  return client as T & { compression: CompressionStats };
}

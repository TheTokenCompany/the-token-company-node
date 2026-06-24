import { TheTokenCompany } from "./client.js";
import { collectToolUseIds, resolveAggressiveness } from "./compress.js";
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

/** Default cap on actual web searches per request (matches the Python SDK). */
export const DEFAULT_WEB_SEARCH_MAX_USES = 10;

/** Resolve the search cap: explicit option > native tool's max_uses > default. */
function effectiveMaxUses(explicit: number | null | undefined, native: number | null): number {
  if (explicit !== null && explicit !== undefined) return explicit;
  if (native !== null) return native;
  return DEFAULT_WEB_SEARCH_MAX_USES;
}

/**
 * Strip Anthropic's server-side web search, inject ttc_web_search, and report
 * the `max_uses` declared on any stripped native `web_search_*` tool (so a
 * caller's existing cap carries over unchanged).
 */
function injectSearchTool(params: any): { params: any; nativeMaxUses: number | null } {
  let nativeMaxUses: number | null = null;
  let tools = params.tools ? [...params.tools] : [];
  tools = tools.filter((t: any) => {
    if (String(t.type ?? "").startsWith("web_search_")) {
      if (nativeMaxUses === null && typeof t.max_uses === "number") {
        nativeMaxUses = t.max_uses;
      }
      return false; // strip the native tool
    }
    return true;
  });
  if (!tools.some((t: any) => t.name === "ttc_web_search")) {
    tools.push(TTC_SEARCH_TOOL);
  }
  return { params: { ...params, tools }, nativeMaxUses };
}

/** Error result for a failed search — native returns one too, so the agent
 * keeps going instead of crashing mid-run. */
function searchFailed(toolUseId: string, err: unknown): any {
  const name = err instanceof Error ? err.constructor.name : "Error";
  const msg = err instanceof Error ? err.message : String(err);
  return {
    type: "tool_result",
    tool_use_id: toolUseId,
    is_error: true,
    content: `Web search failed: ${name}: ${msg}`,
  };
}

/** An error tool_result for a single over-budget search — like the
 * 'max uses exceeded' result a native web-search tool returns. Scoped to THIS
 * query (not a global change): the tool stays available, the model just sees
 * that this search didn't run and answers with what it has. */
function searchBudgetExhausted(toolUseId: string): any {
  return {
    type: "tool_result",
    tool_use_id: toolUseId,
    is_error: true,
    content:
      "This search was not performed: the web-search limit for this request has " +
      "been reached. Do not retry; answer using the results already gathered.",
  };
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
  ttcClient: TheTokenCompany, stats: CompressionStats, rest: any[],
  maxUses: number | null
): Promise<any> {
  const messages = [...(params.messages || [])];
  let searchesDone = 0;
  while (hasSearchToolUse(response)) {
    // Add assistant response
    const assistantContent = response.content.map((b: any) => {
      if (typeof b.toJSON === 'function') return b.toJSON();
      return b;
    });
    messages.push({ role: "assistant", content: assistantContent });

    const searchBlocks = response.content.filter(
      (b: any) => b.type === "tool_use" && b.name === "ttc_web_search"
    );

    // Split this round into searches we can still afford (in order, within the
    // remaining budget) and pre-built budget-exhausted results for the rest.
    const toRun: any[] = [];
    const resultsById: Record<string, any> = {};
    for (const block of searchBlocks) {
      if (maxUses !== null && searchesDone + toRun.length >= maxUses) {
        resultsById[block.id] = searchBudgetExhausted(block.id);
      } else {
        toRun.push(block);
      }
    }

    // Run this round's affordable searches concurrently (native does the same).
    const outcomes = await Promise.allSettled(
      toRun.map((b: any) => ttcClient.search(b.input?.query || ""))
    );
    toRun.forEach((block: any, i: number) => {
      const outcome = outcomes[i];
      if (outcome.status === "fulfilled") {
        stats._recordSearch(outcome.value);
        resultsById[block.id] = {
          type: "tool_result",
          tool_use_id: block.id,
          content: formatSearchResults(outcome.value.results),
        };
      } else {
        resultsById[block.id] = searchFailed(block.id, outcome.reason);
      }
    });
    searchesDone += toRun.length;

    const toolResults = searchBlocks.map((b: any) => resultsById[b.id]);
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
  const model = options.model ?? "bear-2";
  const roleAggr = resolveAggressiveness(options.aggressiveness ?? 0.2);
  if (options.compressAssistant && !("assistant" in roleAggr)) {
    roleAggr["assistant"] = roleAggr["user"] ?? 0.2;
  }
  const stripServerToolResults = options.stripServerToolResults ?? false;
  const webSearch = options.webSearch ?? false;
  const originalCreate = client.messages.create.bind(client.messages);

  client.messages.create = async function (params: any, ...rest: any[]) {
    if (params?.messages) {
      // Messages + system compressed in ONE request; the server walks the
      // roles and tool_result blocks and serves re-sent history from cache.
      const skipToolUseIds = webSearch
        ? [...collectToolUseIds(params.messages, "ttc_web_search")]
        : undefined;
      try {
        const result = await ttcClient.compressChat(params.messages, {
          model,
          format: "anthropic",
          aggressiveness: roleAggr,
          system: params.system,
          stripServerToolResults,
          skipToolUseIds,
        });
        params = { ...params, messages: result.messages };
        if (params.system !== undefined) {
          params = { ...params, system: result.system };
        }
        stats._recordChat(result);
      } catch (e) {
        // Graceful degradation: a compression-backend fault must never break
        // the customer's underlying LLM call. Fall through with the original,
        // uncompressed messages.
        console.warn(`[the-token-company] compression failed (${e}); sending uncompressed.`);
      }
    }

    // Inject search tool (after compression so the injected tool isn't sent
    // to the compressor). Capture any native max_uses to carry it over.
    let nativeMaxUses: number | null = null;
    if (webSearch) {
      const injected = injectSearchTool(params);
      params = injected.params;
      nativeMaxUses = injected.nativeMaxUses;
    }

    let response = await originalCreate(params, ...rest);

    // Handle search tool loop, capped at the resolved max_uses.
    if (webSearch) {
      response = await handleSearchLoop(
        response, params, originalCreate, ttcClient, stats, rest,
        effectiveMaxUses(options.webSearchMaxUses, nativeMaxUses)
      );
    }

    return response;
  } as any;

  (client as any).compression = stats;
  return client as T & { compression: CompressionStats };
}

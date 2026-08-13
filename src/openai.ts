import { TheTokenCompany } from "./client.js";
import { resolveAggressiveness } from "./compress.js";
import type { WithCompressionOptions } from "./types.js";
import { CompressionStats } from "./types.js";

/**
 * Wrap an OpenAI-compatible client to auto-compress conversation turns.
 *
 * Works with `OpenAI`, OpenRouter, and any OpenAI-compatible client.
 *
 * ```ts
 * import OpenAI from "openai";
 * import { withCompression } from "the-token-company/openai";
 *
 * const client = withCompression(new OpenAI(), { compressionApiKey: "ttc-..." });
 * const response = await client.chat.completions.create({
 *   model: "gpt-4o",
 *   messages: [{ role: "user", content: "..." }],
 * });
 * console.log(client.compression.totalTokensSaved);
 * ```
 *
 * Assistant/agent turns are compressed by default. To keep the provider's KV
 * cache warm, pass a per-role `aggressiveness` dict that omits the `assistant`
 * key (e.g. `{ user: 0.2, system: 0.2, tool: 0.2 }`).
 */
export function withCompression<T extends { chat: { completions: { create: Function } } }>(
  client: T,
  options: WithCompressionOptions
): T & { compression: CompressionStats } {
  const stats = new CompressionStats();
  const ttcClient = new TheTokenCompany({ apiKey: options.compressionApiKey, appId: options.appId, fetch: options.fetch });
  const model = options.model ?? "bear-2";
  const roleAggr = resolveAggressiveness(options.aggressiveness ?? 0.2);
  const originalCreate = client.chat.completions.create.bind(client.chat.completions);

  client.chat.completions.create = async function (params: any, ...rest: any[]) {
    if (params?.messages) {
      // One request for the whole conversation — the server walks the roles,
      // compresses every segment concurrently, and serves re-sent history
      // from cache.
      try {
        const result = await ttcClient.compressChat(params.messages, {
          model,
          format: "openai",
          aggressiveness: roleAggr,
        });
        params = { ...params, messages: result.messages };
        stats._recordChat(result);
      } catch (e) {
        // Graceful degradation: a compression-backend fault must never break
        // the customer's underlying LLM call. Send the original messages.
        console.warn(`[the-token-company] compression failed (${e}); sending uncompressed.`);
      }
    }
    return originalCreate(params, ...rest);
  } as any;

  (client as any).compression = stats;
  return client as T & { compression: CompressionStats };
}

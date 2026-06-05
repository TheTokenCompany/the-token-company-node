import { TheTokenCompany } from "./client.js";
import { StatsTTC, compressOpenAIMessages, resolveAggressiveness } from "./compress.js";
import type { WithCompressionOptions } from "./types.js";
import { CompressionStats } from "./types.js";

/**
 * Wrap an OpenAI-compatible client to auto-compress non-assistant messages.
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
 * Assistant messages pass through unchanged to preserve the provider's KV cache.
 */
export function withCompression<T extends { chat: { completions: { create: Function } } }>(
  client: T,
  options: WithCompressionOptions
): T & { compression: CompressionStats } {
  const stats = new CompressionStats();
  const compressor = new StatsTTC(
    new TheTokenCompany({ apiKey: options.compressionApiKey, appId: options.appId }),
    stats
  );
  const model = options.model ?? "bear-2";
  const roleAggr = resolveAggressiveness(options.aggressiveness ?? 0.2);
  const originalCreate = client.chat.completions.create.bind(client.chat.completions);

  client.chat.completions.create = async function (params: any, ...rest: any[]) {
    if (params?.messages) {
      stats._startTurn();
      params = {
        ...params,
        messages: await compressOpenAIMessages(compressor, params.messages, model, roleAggr),
      };
      stats._endTurn();
    }
    return originalCreate(params, ...rest);
  } as any;

  (client as any).compression = stats;
  return client as T & { compression: CompressionStats };
}

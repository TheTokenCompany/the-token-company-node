import { TheTokenCompany } from "./client.js";
import { StatsTTC, compressAnthropicMessages, resolveAggressiveness } from "./compress.js";
import type { WithCompressionOptions } from "./types.js";
import { CompressionStats } from "./types.js";

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
  const compressor = new StatsTTC(
    new TheTokenCompany({ apiKey: options.compressionApiKey, baseUrl: options.baseUrl, appId: options.appId, fetch: options.fetch }),
    stats
  );
  const model = options.model ?? "bear-2";
  const roleAggr = resolveAggressiveness(options.aggressiveness ?? 0.2);
  if (options.compressAssistant && !("assistant" in roleAggr)) {
    roleAggr["assistant"] = roleAggr["user"] ?? 0.2;
  }
  const systemAggr = roleAggr["system"];
  const stripServerToolResults = options.stripServerToolResults ?? false;
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
    stats._endTurn();
    return originalCreate(params, ...rest);
  } as any;

  (client as any).compression = stats;
  return client as T & { compression: CompressionStats };
}

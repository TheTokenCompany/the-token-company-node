import { wrapLanguageModel } from "ai";
import type { LanguageModelMiddleware } from "ai";
import type { LanguageModelV3 } from "@ai-sdk/provider";
import { TheTokenCompany } from "./client.js";
import { StatsTTC, compressAISDKPrompt, resolveAggressiveness } from "./compress.js";
import type { WithCompressionOptions } from "./types.js";
import { CompressionStats } from "./types.js";

/**
 * Create a Vercel AI SDK middleware that auto-compresses non-assistant messages.
 *
 * ```ts
 * import { wrapLanguageModel } from "ai";
 * import { openai } from "@ai-sdk/openai";
 * import { compressionMiddleware } from "the-token-company/ai-sdk";
 *
 * const middleware = compressionMiddleware({ compressionApiKey: "ttc-..." });
 * const model = wrapLanguageModel({ model: openai("gpt-4o"), middleware });
 * // After calls: middleware.compression.totalTokensSaved
 * ```
 */
export function compressionMiddleware(
  options: WithCompressionOptions
): LanguageModelMiddleware & { compression: CompressionStats } {
  const stats = new CompressionStats();
  const compressor = new StatsTTC(
    new TheTokenCompany({ apiKey: options.compressionApiKey, appId: options.appId }),
    stats
  );
  const compressionModel = options.model ?? "bear-2";
  const roleAggr = resolveAggressiveness(options.aggressiveness ?? 0.2);

  const middleware: LanguageModelMiddleware & { compression: CompressionStats } = {
    specificationVersion: "v3",
    compression: stats,
    transformParams: async ({ params }) => {
      if (params.prompt) {
        stats._startTurn();
        const compressed = await compressAISDKPrompt(
          compressor,
          params.prompt as any[],
          compressionModel,
          roleAggr
        );
        stats._endTurn();
        return { ...params, prompt: compressed } as typeof params;
      }
      return params;
    },
  };

  return middleware;
}

/**
 * Wrap any Vercel AI SDK language model with automatic compression.
 *
 * ```ts
 * import { openai } from "@ai-sdk/openai";
 * import { generateText } from "ai";
 * import { withCompression } from "the-token-company/ai-sdk";
 *
 * const model = withCompression(openai("gpt-4o"), { compressionApiKey: "ttc-..." });
 *
 * const { text } = await generateText({
 *   model,
 *   messages: [{ role: "user", content: "Summarize these results..." }],
 * });
 * console.log(model.compression.totalTokensSaved);
 * ```
 *
 * Works with any provider (`@ai-sdk/openai`, `@ai-sdk/anthropic`, `@ai-sdk/google`, etc.).
 */
export function withCompression(
  model: LanguageModelV3,
  options: WithCompressionOptions
): LanguageModelV3 & { compression: CompressionStats } {
  const mw = compressionMiddleware(options);
  const wrapped = wrapLanguageModel({ model, middleware: mw }) as LanguageModelV3 & {
    compression: CompressionStats;
  };
  wrapped.compression = mw.compression;
  return wrapped;
}

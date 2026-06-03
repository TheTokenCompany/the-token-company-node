import { TheTokenCompany } from "./client.js";
import type { Aggressiveness, CompressResult } from "./types.js";
import { CompressionStats } from "./types.js";

const DEFAULT_ROLES = ["user", "system", "tool"] as const;

export interface Compressor {
  compress(text: string, options: { model?: string; aggressiveness?: number }): Promise<CompressResult>;
}

export class AnalyticsTTC implements Compressor {
  constructor(
    private readonly inner: TheTokenCompany,
    public readonly stats: CompressionStats
  ) {}

  async compress(text: string, options: { model?: string; aggressiveness?: number }): Promise<CompressResult> {
    const result = await this.inner.compress(text, options);
    this.stats._record(result);
    return result;
  }
}

export function resolveAggressiveness(aggressiveness: Aggressiveness): Record<string, number> {
  if (typeof aggressiveness === "number") {
    return Object.fromEntries(DEFAULT_ROLES.map((r) => [r, aggressiveness]));
  }
  return aggressiveness;
}

// ---------------------------------------------------------------------------
// OpenAI message compression
// ---------------------------------------------------------------------------

interface OpenAIMessage {
  role: string;
  content?: string | Array<{ type: string; text?: string; [key: string]: unknown }> | null;
  [key: string]: unknown;
}

const OPENAI_TOOL_ROLES = new Set(["tool", "function"]);

function openaiAggr(role: string, roleAggr: Record<string, number>): number | undefined {
  if (OPENAI_TOOL_ROLES.has(role)) return roleAggr["tool"];
  return roleAggr[role];
}

export async function compressOpenAIMessages(
  ttc: Compressor,
  messages: OpenAIMessage[],
  model: string,
  roleAggr: Record<string, number>
): Promise<OpenAIMessage[]> {
  return Promise.all(
    messages.map(async (msg) => {
      const aggr = openaiAggr(msg.role, roleAggr);
      if (aggr == null) return msg;

      if (typeof msg.content === "string" && msg.content.trim()) {
        const result = await ttc.compress(msg.content, { model, aggressiveness: aggr });
        return { ...msg, content: result.output };
      }

      if (Array.isArray(msg.content)) {
        const blocks = await Promise.all(
          msg.content.map(async (block) => {
            if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
              const result = await ttc.compress(block.text, { model, aggressiveness: aggr });
              return { ...block, text: result.output };
            }
            return block;
          })
        );
        return { ...msg, content: blocks };
      }

      return msg;
    })
  );
}

// ---------------------------------------------------------------------------
// Anthropic message compression
// ---------------------------------------------------------------------------

interface AnthropicBlock {
  type: string;
  text?: string;
  content?: string | Array<{ type: string; text?: string; [key: string]: unknown }>;
  [key: string]: unknown;
}

interface AnthropicMessage {
  role: string;
  content?: string | AnthropicBlock[] | null;
  [key: string]: unknown;
}

export async function compressAnthropicMessages(
  ttc: Compressor,
  messages: AnthropicMessage[],
  model: string,
  roleAggr: Record<string, number>
): Promise<AnthropicMessage[]> {
  return Promise.all(
    messages.map(async (msg) => {
      if (msg.role !== "user") return msg;

      const userAggr = roleAggr["user"];
      const toolAggr = roleAggr["tool"];
      if (userAggr == null && toolAggr == null) return msg;

      if (typeof msg.content === "string" && userAggr != null && msg.content.trim()) {
        const result = await ttc.compress(msg.content, { model, aggressiveness: userAggr });
        return { ...msg, content: result.output };
      }

      if (Array.isArray(msg.content)) {
        const blocks = await compressAnthropicBlocks(ttc, msg.content, model, userAggr, toolAggr);
        return { ...msg, content: blocks };
      }

      return msg;
    })
  );
}

async function compressAnthropicBlocks(
  ttc: Compressor,
  blocks: AnthropicBlock[],
  model: string,
  userAggr: number | undefined,
  toolAggr: number | undefined
): Promise<AnthropicBlock[]> {
  return Promise.all(
    blocks.map(async (block) => {
      if (block.type === "text" && userAggr != null && typeof block.text === "string" && block.text.trim()) {
        const result = await ttc.compress(block.text, { model, aggressiveness: userAggr });
        return { ...block, text: result.output };
      }

      if (block.type === "tool_result" && toolAggr != null) {
        return compressToolResult(ttc, block, model, toolAggr);
      }

      return block;
    })
  );
}

async function compressToolResult(
  ttc: Compressor,
  block: AnthropicBlock,
  model: string,
  aggressiveness: number
): Promise<AnthropicBlock> {
  if (typeof block.content === "string" && block.content.trim()) {
    const result = await ttc.compress(block.content, { model, aggressiveness });
    return { ...block, content: result.output };
  }
  if (Array.isArray(block.content)) {
    const compressed = await Promise.all(
      block.content.map(async (inner) => {
        if (inner.type === "text" && typeof inner.text === "string" && inner.text.trim()) {
          const result = await ttc.compress(inner.text, { model, aggressiveness });
          return { ...inner, text: result.output };
        }
        return inner;
      })
    );
    return { ...block, content: compressed };
  }
  return block;
}

// ---------------------------------------------------------------------------
// Vercel AI SDK message compression
// ---------------------------------------------------------------------------

interface AISDKTextPart {
  type: "text";
  text: string;
  [key: string]: unknown;
}

interface AISDKToolResultPart {
  type: "tool-result";
  toolCallId: string;
  toolName: string;
  result: unknown;
  [key: string]: unknown;
}

type AISDKPart = AISDKTextPart | AISDKToolResultPart | { type: string; [key: string]: unknown };

interface AISDKMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | AISDKPart[];
  [key: string]: unknown;
}

export async function compressAISDKPrompt(
  ttc: Compressor,
  prompt: AISDKMessage[],
  model: string,
  roleAggr: Record<string, number>
): Promise<AISDKMessage[]> {
  return Promise.all(
    prompt.map(async (msg) => {
      if (msg.role === "assistant") return msg;

      if (msg.role === "system") {
        const aggr = roleAggr["system"];
        if (aggr == null) return msg;
        if (typeof msg.content === "string" && msg.content.trim()) {
          const result = await ttc.compress(msg.content, { model, aggressiveness: aggr });
          return { ...msg, content: result.output };
        }
        return msg;
      }

      if (msg.role === "user") {
        const userAggr = roleAggr["user"];
        if (userAggr == null) return msg;
        if (typeof msg.content === "string" && msg.content.trim()) {
          const result = await ttc.compress(msg.content, { model, aggressiveness: userAggr });
          return { ...msg, content: result.output };
        }
        if (Array.isArray(msg.content)) {
          const parts = await Promise.all(
            msg.content.map(async (part) => {
              if (part.type === "text" && "text" in part && typeof part.text === "string" && part.text.trim()) {
                const result = await ttc.compress(part.text, { model, aggressiveness: userAggr });
                return { ...part, text: result.output };
              }
              return part;
            })
          );
          return { ...msg, content: parts };
        }
        return msg;
      }

      if (msg.role === "tool") {
        const toolAggr = roleAggr["tool"];
        if (toolAggr == null) return msg;
        if (Array.isArray(msg.content)) {
          const parts = await Promise.all(
            msg.content.map(async (part) => {
              if (part.type === "tool-result" && "result" in part && typeof part.result === "string" && part.result.trim()) {
                const result = await ttc.compress(part.result, { model, aggressiveness: toolAggr });
                return { ...part, result: result.output };
              }
              return part;
            })
          );
          return { ...msg, content: parts };
        }
        return msg;
      }

      return msg;
    })
  );
}

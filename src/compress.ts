import { TheTokenCompany } from "./client.js";
import type { Aggressiveness, CompressResult } from "./types.js";
import { CompressionStats } from "./types.js";

// Assistant/agent turns are compressed by default alongside user/system/tool.
// To keep the provider's KV cache warm, pass a per-role aggressiveness dict that
// omits the "assistant" key (see resolveAggressiveness).
const DEFAULT_ROLES = ["user", "system", "tool", "assistant"] as const;
const SERVER_TOOL_BLOCK_TYPES = new Set(["web_search_tool_result", "server_tool_use"]);

export interface Compressor {
  compress(text: string, options: { model?: string; aggressiveness?: number }): Promise<CompressResult>;
}

export class StatsTTC implements Compressor {
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
  tool_use_id?: string;
  content?: string | Array<{ type: string; text?: string; [key: string]: unknown }>;
  [key: string]: unknown;
}

interface AnthropicMessage {
  role: string;
  content?: string | AnthropicBlock[] | null;
  [key: string]: unknown;
}

export function collectToolUseIds(messages: AnthropicMessage[], toolName: string): Set<string> {
  const ids = new Set<string>();
  for (const msg of messages) {
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === "tool_use" && block.name === toolName && block.id) {
        ids.add(block.id as string);
      }
    }
  }
  return ids;
}

export async function compressAnthropicMessages(
  ttc: Compressor,
  messages: AnthropicMessage[],
  model: string,
  roleAggr: Record<string, number>,
  options?: { stripServerToolResults?: boolean; skipToolName?: string }
): Promise<AnthropicMessage[]> {
  const stripServerToolResults = options?.stripServerToolResults ?? false;
  const skipIds = options?.skipToolName ? collectToolUseIds(messages, options.skipToolName) : undefined;
  return Promise.all(
    messages.map(async (msg) => {
      if (msg.role === "assistant") {
        const assistantAggr = roleAggr["assistant"];
        if (assistantAggr == null && !stripServerToolResults) return msg;

        if (typeof msg.content === "string" && assistantAggr != null && msg.content.trim()) {
          const result = await ttc.compress(msg.content, { model, aggressiveness: assistantAggr });
          return { ...msg, content: result.output };
        }

        if (Array.isArray(msg.content)) {
          const blocks = await compressAssistantBlocks(ttc, msg.content, model, assistantAggr, stripServerToolResults);
          return { ...msg, content: blocks };
        }

        return msg;
      }

      if (msg.role !== "user") return msg;

      const userAggr = roleAggr["user"];
      const toolAggr = roleAggr["tool"];
      if (userAggr == null && toolAggr == null) return msg;

      if (typeof msg.content === "string" && userAggr != null && msg.content.trim()) {
        const result = await ttc.compress(msg.content, { model, aggressiveness: userAggr });
        return { ...msg, content: result.output };
      }

      if (Array.isArray(msg.content)) {
        const blocks = await compressAnthropicBlocks(ttc, msg.content, model, userAggr, toolAggr, skipIds);
        return { ...msg, content: blocks };
      }

      return msg;
    })
  );
}

async function compressAssistantBlocks(
  ttc: Compressor,
  blocks: AnthropicBlock[],
  model: string,
  assistantAggr: number | undefined,
  stripServerToolResults: boolean
): Promise<AnthropicBlock[]> {
  const results = await Promise.all(
    blocks.map(async (block) => {
      if (stripServerToolResults && SERVER_TOOL_BLOCK_TYPES.has(block.type)) {
        return null;
      }
      if (block.type === "text" && assistantAggr != null && typeof block.text === "string" && block.text.trim()) {
        const result = await ttc.compress(block.text, { model, aggressiveness: assistantAggr });
        return { ...block, text: result.output };
      }
      return block;
    })
  );
  const filtered = results.filter((b): b is AnthropicBlock => b != null);
  return filtered.length > 0 ? filtered : blocks;
}

async function compressAnthropicBlocks(
  ttc: Compressor,
  blocks: AnthropicBlock[],
  model: string,
  userAggr: number | undefined,
  toolAggr: number | undefined,
  skipIds?: Set<string>
): Promise<AnthropicBlock[]> {
  return Promise.all(
    blocks.map(async (block) => {
      if (block.type === "text" && userAggr != null && typeof block.text === "string" && block.text.trim()) {
        const result = await ttc.compress(block.text, { model, aggressiveness: userAggr });
        return { ...block, text: result.output };
      }

      if (block.type === "tool_result" && toolAggr != null) {
        if (block.tool_use_id && skipIds?.has(block.tool_use_id)) return block;
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
      if (msg.role === "assistant") {
        const assistantAggr = roleAggr["assistant"];
        if (assistantAggr == null) return msg;
        if (typeof msg.content === "string" && msg.content.trim()) {
          const result = await ttc.compress(msg.content, { model, aggressiveness: assistantAggr });
          return { ...msg, content: result.output };
        }
        if (Array.isArray(msg.content)) {
          const parts = await Promise.all(
            msg.content.map(async (part) => {
              if (part.type === "text" && "text" in part && typeof part.text === "string" && part.text.trim()) {
                const result = await ttc.compress(part.text, { model, aggressiveness: assistantAggr });
                return { ...part, text: result.output };
              }
              return part;
            })
          );
          return { ...msg, content: parts };
        }
        return msg;
      }

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

# The Token Company Node.js SDK

Compress LLM prompts to reduce costs and latency. 100K tokens compressed in ~85ms.

[![npm version](https://img.shields.io/npm/v/the-token-company)](https://www.npmjs.com/package/the-token-company)
[![npm downloads](https://img.shields.io/npm/dm/the-token-company)](https://www.npmjs.com/package/the-token-company)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TheTokenCompany/the-token-company-node/blob/main/LICENSE)

[Docs](https://thetokencompany.com/docs) · [Website](https://thetokencompany.com) · [Dashboard](https://app.thetokencompany.com) · [Python SDK](https://github.com/TheTokenCompany/the-token-company-python)

## Install

```bash
npm install the-token-company
```

## Quick start

```ts
import { TheTokenCompany } from "the-token-company";

const client = new TheTokenCompany({ apiKey: "ttc-..." });
const result = await client.compress("Your long prompt text here...", { model: "bear-2" });

console.log(result.output);          // compressed text
console.log(result.tokensSaved);     // tokens removed
console.log(result.compressionRatio); // e.g. 1.8
```

## SDK wrappers

Drop-in wrappers that auto-compress all non-assistant messages before sending to your LLM. Assistant messages pass through unchanged so the provider's KV cache stays warm.

### OpenAI / OpenRouter

```ts
import OpenAI from "openai";
import { withCompression } from "the-token-company/openai";

const client = withCompression(new OpenAI(), { compressionApiKey: "ttc-..." });

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant..." },
    { role: "user", content: "Summarize these results..." },
  ],
});
```

For OpenRouter, just set the base URL:

```ts
const client = withCompression(
  new OpenAI({ baseURL: "https://openrouter.ai/api/v1", apiKey: "or-..." }),
  { compressionApiKey: "ttc-..." }
);
```

### Anthropic

```ts
import Anthropic from "@anthropic-ai/sdk";
import { withCompression } from "the-token-company/anthropic";

const client = withCompression(new Anthropic(), { compressionApiKey: "ttc-..." });

const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 1024,
  system: "You are a helpful assistant...",
  messages: [{ role: "user", content: "Summarize these results..." }],
});
```

Both `messages` and the `system` parameter are compressed.

### Vercel AI SDK

**`withCompression()` one-liner** — wraps any AI SDK model with automatic compression:

```ts
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import { withCompression } from "the-token-company/ai-sdk";

const model = withCompression(openai("gpt-4o"), { compressionApiKey: "ttc-..." });

const { text } = await generateText({
  model,
  messages: [{ role: "user", content: "Summarize these results..." }],
});
```

Works with any provider (`@ai-sdk/openai`, `@ai-sdk/anthropic`, `@ai-sdk/google`, etc.).

**`compressionMiddleware()` for composition** — use when combining with other middleware:

```ts
import { wrapLanguageModel, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { compressionMiddleware } from "the-token-company/ai-sdk";

const model = wrapLanguageModel({
  model: openai("gpt-4o"),
  middleware: compressionMiddleware({ compressionApiKey: "ttc-..." }),
});
```

## Models

| Model      | Description            |
|------------|------------------------|
| `bear-2`   | Latest, recommended    |
| `bear-1.2` | Previous generation    |
| `bear-1.1` | Legacy                 |
| `bear-1`   | Legacy                 |

## Aggressiveness

Control compression intensity — a single number applies to all roles, or pass a per-role object:

```ts
// All roles at 0.5
withCompression(client, { compressionApiKey: "ttc-...", aggressiveness: 0.5 });

// Per-role — only listed roles are compressed
withCompression(client, {
  compressionApiKey: "ttc-...",
  aggressiveness: { system: 0.1, user: 0.3, tool: 0.5 },
});
```

| Role key   | OpenAI                          | Anthropic                      | AI SDK              |
|------------|---------------------------------|--------------------------------|---------------------|
| `user`     | `role: "user"` messages         | User text content              | User messages       |
| `system`   | `role: "system"` messages       | `system` parameter             | System messages     |
| `tool`     | `tool` + `function` messages    | `tool_result` content blocks   | Tool result parts   |

## Gzip

Gzip compression of request payloads is on by default. Disable with:

```ts
const client = new TheTokenCompany({ apiKey: "ttc-...", gzip: false });
```

## Response

`CompressResult` fields:

| Field              | Type     | Description                        |
|--------------------|----------|------------------------------------|
| `output`           | `string` | Compressed text                    |
| `outputTokens`     | `number` | Token count after compression      |
| `inputTokens`      | `number` | Token count before compression     |
| `tokensSaved`      | `number` | Tokens removed                     |
| `compressionRatio` | `number` | Ratio (e.g. 1.8x)                 |

## License

MIT

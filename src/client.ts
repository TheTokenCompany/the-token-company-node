import { gzip } from "node:zlib";
import { promisify } from "node:util";
import type { CompressRequest, CompressResponse, CompressResult, TheTokenCompanyOptions } from "./types.js";
import {
  APIError,
  AuthenticationError,
  InvalidRequestError,
  PaymentRequiredError,
  RateLimitError,
  RequestTooLargeError,
} from "./errors.js";

const gzipAsync = promisify(gzip);

const BASE_URL = "https://api.thetokencompany.com";
const DEFAULT_TIMEOUT = 30_000;

export const BEAR_1 = "bear-1";
export const BEAR_1_1 = "bear-1.1";
export const BEAR_1_2 = "bear-1.2";
export const BEAR_2 = "bear-2";

export class TheTokenCompany {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeout: number;
  private readonly gzip: boolean;
  private readonly appId?: string;
  private readonly fetchFn: typeof globalThis.fetch;

  constructor(options: TheTokenCompanyOptions) {
    this.apiKey = options.apiKey;
    this.baseUrl = (options.baseUrl ?? BASE_URL).replace(/\/$/, "");
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.gzip = options.gzip ?? true;
    this.appId = options.appId;
    this.fetchFn = options.fetch ?? globalThis.fetch;
  }

  async compress(
    text: string,
    options: { model?: string; aggressiveness?: number; appId?: string } = {}
  ): Promise<CompressResult> {
    const { model = "bear-2", aggressiveness = 0.2 } = options;

    if (!text || !text.trim()) {
      throw new InvalidRequestError("text cannot be empty");
    }
    if (aggressiveness < 0 || aggressiveness > 1) {
      throw new InvalidRequestError("aggressiveness must be between 0.0 and 1.0");
    }

    const resolvedAppId = options.appId ?? this.appId;
    const payload: CompressRequest = {
      model,
      input: text,
      compression_settings: { aggressiveness },
      ...(resolvedAppId !== undefined && { app_id: resolvedAppId }),
    };

    const jsonBody = JSON.stringify(payload);
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
    };

    let body: Uint8Array | string;
    if (this.gzip) {
      headers["Content-Encoding"] = "gzip";
      body = new Uint8Array(await gzipAsync(jsonBody));
    } else {
      body = jsonBody;
    }

    const response = await this.fetchFn(`${this.baseUrl}/v1/compress`, {
      method: "POST",
      headers,
      body: body as BodyInit,
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      throw await this.parseError(response);
    }

    const data = (await response.json()) as CompressResponse;
    return {
      output: data.output,
      outputTokens: data.output_tokens,
      inputTokens: data.original_input_tokens,
      tokensSaved: data.original_input_tokens - data.output_tokens,
      compressionRatio:
        data.output_tokens === 0 ? 0 : data.original_input_tokens / data.output_tokens,
    };
  }

  private async parseError(response: Response): Promise<Error> {
    let msg: string;
    try {
      const body = (await response.json()) as { detail?: string | { message?: string } };
      const detail = body.detail ?? "Unknown error";
      msg = typeof detail === "object" ? (detail.message ?? JSON.stringify(detail)) : detail;
    } catch {
      msg = (await response.text()) || "Unknown error";
    }

    switch (response.status) {
      case 401:
        return new AuthenticationError(msg);
      case 402:
        return new PaymentRequiredError(msg);
      case 413:
        return new RequestTooLargeError(msg);
      case 429:
        return new RateLimitError(msg);
      case 400:
        return new InvalidRequestError(msg);
      default:
        return new APIError(`API error (${response.status}): ${msg}`, response.status);
    }
  }
}

export interface CompressRequest {
  model: string;
  input: string;
  compression_settings: {
    aggressiveness: number;
  };
  app_id?: string;
}

export interface CompressResponse {
  output: string;
  output_tokens: number;
  original_input_tokens: number;
}

export interface CompressResult {
  output: string;
  outputTokens: number;
  inputTokens: number;
  tokensSaved: number;
  compressionRatio: number;
}

export interface TheTokenCompanyOptions {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  gzip?: boolean;
  appId?: string;
  fetch?: typeof globalThis.fetch;
}

export type Aggressiveness = number | Record<string, number>;

export interface WithCompressionOptions {
  compressionApiKey: string;
  model?: string;
  aggressiveness?: Aggressiveness;
  /** Compress text blocks in assistant messages (multi-turn web search optimization). */
  compressAssistant?: boolean;
  /** Strip server-side tool result blocks (e.g. web_search_tool_result) from assistant messages. Disables citations in subsequent turns. */
  stripServerToolResults?: boolean;
  baseUrl?: string;
  appId?: string;
  fetch?: typeof globalThis.fetch;
}

export interface SearchRequestOptions {
  maxResults?: number;
  searchDepth?: "basic" | "advanced";
  includeRawContent?: boolean;
  model?: string;
  aggressiveness?: number;
  appId?: string;
}

export interface SearchResultItem {
  url: string;
  title: string;
  content: string;
  score?: number;
}

export interface SearchResult {
  results: SearchResultItem[];
  query: string;
  searchTime: number;
  originalInputTokens: number;
  outputTokens: number;
  tokensSaved: number;
}

export interface TurnStats {
  inputTokens: number;
  outputTokens: number;
  tokensSaved: number;
  messagesCompressed: number;
  ratio: number;
  timestamp: number;
}

export class CompressionStats {
  history: TurnStats[] = [];
  private _accumulator: CompressResult[] = [];

  _startTurn(): void {
    this._accumulator = [];
  }

  _record(result: CompressResult): void {
    this._accumulator.push(result);
  }

  _endTurn(): void {
    if (this._accumulator.length > 0) {
      const inputTokens = this._accumulator.reduce((s, r) => s + r.inputTokens, 0);
      const outputTokens = this._accumulator.reduce((s, r) => s + r.outputTokens, 0);
      this.history.push({
        inputTokens,
        outputTokens,
        tokensSaved: inputTokens - outputTokens,
        messagesCompressed: this._accumulator.length,
        ratio: outputTokens === 0 ? 0 : inputTokens / outputTokens,
        timestamp: Date.now(),
      });
    }
    this._accumulator = [];
  }

  get totalInputTokens(): number {
    return this.history.reduce((s, t) => s + t.inputTokens, 0);
  }

  get totalOutputTokens(): number {
    return this.history.reduce((s, t) => s + t.outputTokens, 0);
  }

  get totalTokensSaved(): number {
    return this.history.reduce((s, t) => s + t.tokensSaved, 0);
  }

  get calls(): number {
    return this.history.length;
  }

  get ratio(): number {
    return this.totalOutputTokens === 0 ? 0 : this.totalInputTokens / this.totalOutputTokens;
  }
}

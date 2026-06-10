export { TheTokenCompany, BEAR_1, BEAR_1_1, BEAR_1_2, BEAR_2 } from "./client.js";
export {
  TheTokenCompanyError,
  AuthenticationError,
  InvalidRequestError,
  PaymentRequiredError,
  RequestTooLargeError,
  RateLimitError,
  APIError,
} from "./errors.js";
export { CompressionStats } from "./types.js";
export type {
  CompressResult,
  TheTokenCompanyOptions,
  Aggressiveness,
  WithCompressionOptions,
  TurnStats,
  SearchRequestOptions,
  SearchResultItem,
  SearchResult,
} from "./types.js";

/**
 * Wrap text in `<ttc_safe>` tags to protect it from compression.
 */
export function protect(text: string): string {
  return `<ttc_safe>${text}</ttc_safe>`;
}

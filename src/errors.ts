export class TheTokenCompanyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TheTokenCompanyError";
  }
}

export class AuthenticationError extends TheTokenCompanyError {
  constructor(message: string) {
    super(message);
    this.name = "AuthenticationError";
  }
}

export class InvalidRequestError extends TheTokenCompanyError {
  constructor(message: string) {
    super(message);
    this.name = "InvalidRequestError";
  }
}

export class PaymentRequiredError extends TheTokenCompanyError {
  constructor(message: string) {
    super(message);
    this.name = "PaymentRequiredError";
  }
}

export class RequestTooLargeError extends TheTokenCompanyError {
  constructor(message: string) {
    super(message);
    this.name = "RequestTooLargeError";
  }
}

export class RateLimitError extends TheTokenCompanyError {
  constructor(message: string) {
    super(message);
    this.name = "RateLimitError";
  }
}

export class APIError extends TheTokenCompanyError {
  statusCode: number | undefined;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "APIError";
    this.statusCode = statusCode;
  }
}

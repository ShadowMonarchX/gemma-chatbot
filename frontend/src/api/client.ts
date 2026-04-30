export class ApiError extends Error {
  public status: number;

  public requestId: string;

  public constructor(status: number, requestId: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.requestId = requestId;
  }
}

export class ApiClient {
  private baseUrl: string;

  public constructor(baseUrl: string = ApiClient.resolveBaseUrl()) {
    this.baseUrl = baseUrl;
  }

  private static resolveBaseUrl(): string {
    const envBase =
      typeof import.meta !== 'undefined' &&
      import.meta.env &&
      typeof import.meta.env.VITE_API_BASE_URL === 'string'
        ? import.meta.env.VITE_API_BASE_URL
        : '';

    if (envBase.trim().length > 0) {
      return envBase.replace(/\/+$/, '');
    }

    return 'http://127.0.0.1:8000/api';
  }

  public async apiCall<T>(url: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
    });

    if (!response.ok) {
      throw await this.buildApiError(response);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    const payload = (await response.json()) as T;
    return payload;
  }

  public async apiStream(url: string, options: RequestInit = {}): Promise<Response> {
    const response = await fetch(`${this.baseUrl}${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
    });

    if (!response.ok) {
      throw await this.buildApiError(response);
    }

    return response;
  }

  private async buildApiError(response: Response): Promise<ApiError> {
    const requestId =
      response.headers.get('X-Request-Id') ??
      response.headers.get('x-request-id') ??
      'unknown-request-id';

    let message = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as {
        detail?: string;
        error?: string;
        message?: string;
      };
      message = payload.error ?? payload.detail ?? payload.message ?? message;
    } catch {
      message = response.statusText || message;
    }

    return new ApiError(response.status, requestId, message);
  }
}

const defaultClient = new ApiClient();

export async function apiCall<T>(url: string, options: RequestInit = {}): Promise<T> {
  return defaultClient.apiCall<T>(url, options);
}

export async function apiStream(url: string, options: RequestInit = {}): Promise<Response> {
  return defaultClient.apiStream(url, options);
}

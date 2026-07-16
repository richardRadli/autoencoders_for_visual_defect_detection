export const DATA_OPS = "/data-ops"
export const DEFECT = "/defect"

export type QueryParams = Record<
  string,
  string | number | boolean | undefined
>

/* Thrown when an API request receives a non-successful HTTP response.
   Service health is checked separately because a proxy may return 500/502
   when the target service is unavailable. */
export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError
}

export function isAbortError(error: unknown): boolean {
  return (
    (typeof DOMException !== "undefined" &&
      error instanceof DOMException &&
      error.name === "AbortError") ||
    (error instanceof Error && error.name === "AbortError")
  )
}

export function buildQuery(params: QueryParams): string {
  const search = new URLSearchParams()

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") {
      search.append(key, String(value))
    }
  }

  const query = search.toString()
  return query ? `?${query}` : ""
}

export function buildUrl(
  base: string,
  path: string,
  params?: QueryParams,
): string {
  return `${base}${path}${params ? buildQuery(params) : ""}`
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(path, options)

  if (!response.ok) {
    let detail = response.statusText

    try {
      const body = await response.json()

      if (body?.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail)
      }
    } catch {
      // The response has no JSON error body.
    }

    throw new ApiError(detail, response.status)
  }

  return response.json() as Promise<T>
}

export function apiGet<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(path, { signal })
}

export function apiPost<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(path, {
    method: "POST",
    signal,
  })
}

export async function checkServiceHealth(
  path: string,
  signal?: AbortSignal,
): Promise<boolean> {
  try {
    const response = await fetch(path, {
      method: "GET",
      signal,
      cache: "no-store",
    })

    return response.ok
  } catch (error) {
    if (isAbortError(error)) {
      throw error
    }

    return false
  }
}
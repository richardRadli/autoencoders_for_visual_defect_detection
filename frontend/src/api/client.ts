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

function humanizeFieldName(value: unknown): string | null {
  if (typeof value !== "string") {
    return null
  }

  const normalized = value.replace(/_/g, " ").trim()

  if (normalized === "") {
    return null
  }

  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

function formatValidationItem(item: unknown): string | null {
  if (typeof item !== "object" || item === null) {
    return null
  }

  const validationError = item as Record<string, unknown>

  if (typeof validationError.msg !== "string") {
    return null
  }

  const location = Array.isArray(validationError.loc)
    ? validationError.loc
    : []

  const field =
    location.length > 0
      ? humanizeFieldName(location[location.length - 1])
      : null

  return field
    ? `${field}: ${validationError.msg}`
    : validationError.msg
}

function formatApiDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") {
    return detail
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map(formatValidationItem)
      .filter((message): message is string => message !== null)

    if (messages.length > 0) {
      return messages.join("; ")
    }
  }

  if (detail !== undefined && detail !== null) {
    try {
      const serialized = JSON.stringify(detail)

      if (serialized) {
        return serialized
      }
    } catch {
      // Fall back to the HTTP status text.
    }
  }

  return fallback
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(path, options)

  if (!response.ok) {
    let detail = response.statusText

    try {
      const body = (await response.json()) as unknown

      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body
      ) {
        detail = formatApiDetail(
          (body as Record<string, unknown>).detail,
          detail,
        )
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
const request = async <T>(path: string, options?: RequestInit): Promise<T> => {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      // no JSON body
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const DATA_OPS = "/data-ops"
export const DEFECT = "/defect"

export const apiGet = <T>(path: string) => request<T>(path)
export const apiPost = <T>(path: string) => request<T>(path, { method: "POST" })

export const buildQuery = (
  params: Record<string, string | number | boolean | undefined>,
): string => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") {
      search.append(key, String(value))
    }
  }
  const query = search.toString()
  return query ? `?${query}` : ""
}
import { useCallback, useEffect, useState } from "react"

import { isAbortError } from "../api/client"
import type { DatasetType } from "../api/types"

export type Readiness<T> = {
  data: T | null
  loading: boolean
  error: string | null
  reload: () => void
}

/*
 * Loads readiness facts for one dataset. Generic over the response because the
 * two services report different shapes: data_operations answers with
 * augmentation/draw_rectangles, defect_detection with aug/noise plus
 * trained_networks.
 *
 * Pass a module-level fetcher (getOpsReadiness / getDefectReadiness) so the
 * effect is not re-run on every render.
 */
export function useReadiness<T>(
  fetcher: (datasetType: DatasetType, signal?: AbortSignal) => Promise<T>,
  datasetType: DatasetType,
): Readiness<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false

    // Cleared up front so a slow answer can never render against the dataset
    // that was selected before it.
    setData(null)
    setError(null)
    setLoading(true)

    fetcher(datasetType, controller.signal)
      .then((res) => {
        if (cancelled) {
          return
        }
        setData(res)
        setLoading(false)
      })
      .catch((e) => {
        if (cancelled || isAbortError(e)) {
          return
        }
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [fetcher, datasetType, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])

  return { data, loading, error, reload }
}
import { useCallback, useEffect, useState } from "react"

import { isAbortError } from "../api/client"
import { getDeviceStatus } from "../api/defectDetection"
import type { DeviceStatus } from "../api/types"

export type DeviceStatusResult = {
  data: DeviceStatus | null
  loading: boolean
  error: string | null
  reload: () => void
}

export function useDeviceStatus(
  intervalMs = 10_000,
): DeviceStatusResult {
  const [data, setData] = useState<DeviceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let timer: number | undefined

    setLoading(true)
    setError(null)

    const load = async () => {
      try {
        const response = await getDeviceStatus(controller.signal)

        if (cancelled) {
          return
        }

        setData(response)
        setError(null)
        setLoading(false)
      } catch (loadError) {
        if (cancelled || isAbortError(loadError)) {
          return
        }

        setData(null)
        setError(
          loadError instanceof Error
            ? loadError.message
            : String(loadError),
        )
        setLoading(false)
      }

      if (!cancelled) {
        timer = window.setTimeout(load, intervalMs)
      }
    }

    void load()

    return () => {
      cancelled = true
      controller.abort()

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [intervalMs, nonce])

  const reload = useCallback(() => {
    setNonce((current) => current + 1)
  }, [])

  return {
    data,
    loading,
    error,
    reload,
  }
}
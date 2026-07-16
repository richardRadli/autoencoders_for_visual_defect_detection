import { useCallback, useEffect, useState } from "react"

import {
  DATA_OPS,
  DEFECT,
  checkServiceHealth,
  isAbortError,
} from "../api/client"
import type { ServiceId } from "../config/workflow"

export type HealthState = "checking" | "online" | "offline"

export type ServiceHealth = Record<ServiceId, HealthState>

export type ServiceHealthResult = {
  health: ServiceHealth
  refresh: () => void
}

const CHECKING: ServiceHealth = {
  data_operations: "checking",
  defect_detection: "checking",
}

export function useServiceHealth(
  intervalMs = 10_000,
): ServiceHealthResult {
  const [health, setHealth] = useState<ServiceHealth>(CHECKING)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let timer: number | undefined

    setHealth(CHECKING)

    const check = async () => {
      try {
        const [dataOperationsOnline, defectDetectionOnline] =
          await Promise.all([
            checkServiceHealth(
              `${DATA_OPS}/openapi.json`,
              controller.signal,
            ),
            checkServiceHealth(
              `${DEFECT}/openapi.json`,
              controller.signal,
            ),
          ])

        if (cancelled) {
          return
        }

        setHealth({
          data_operations: dataOperationsOnline
            ? "online"
            : "offline",
          defect_detection: defectDetectionOnline
            ? "online"
            : "offline",
        })
      } catch (error) {
        if (cancelled || isAbortError(error)) {
          return
        }

        setHealth({
          data_operations: "offline",
          defect_detection: "offline",
        })
      }

      if (!cancelled) {
        timer = window.setTimeout(check, intervalMs)
      }
    }

    void check()

    return () => {
      cancelled = true
      controller.abort()

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [intervalMs, nonce])

  const refresh = useCallback(() => {
    setNonce((current) => current + 1)
  }, [])

  return {
    health,
    refresh,
  }
}
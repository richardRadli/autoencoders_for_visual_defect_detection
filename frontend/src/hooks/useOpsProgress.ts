import { useEffect, useState } from "react"

import { isAbortError } from "../api/client"
import type { OpsProgress } from "../api/types"

/*
 * Polls a data_operations progress endpoint while a blocking run is active.
 *
 * The next request is scheduled only after the previous one settles, so
 * requests never overlap; the in-flight request is aborted on unmount (and
 * whenever polling stops). Returns the latest {running, processed, total}, or
 * null before the first response.
 *
 * Pass a stable module-level fetcher (getAugmentationProgress /
 * getDrawRectanglesProgress) so the effect is not restarted on every render.
 */
export function useOpsProgress(
  fetcher: (signal?: AbortSignal) => Promise<OpsProgress>,
  active: boolean,
  intervalMs = 1000,
): OpsProgress | null {
  const [progress, setProgress] = useState<OpsProgress | null>(null)

  useEffect(() => {
    if (!active) {
      return
    }

    let cancelled = false
    let timer: number | undefined
    const controller = new AbortController()

    const poll = async () => {
      try {
        const next = await fetcher(controller.signal)

        if (cancelled) {
          return
        }

        setProgress(next)
      } catch (error) {
        if (cancelled || isAbortError(error)) {
          return
        }
        // Progress is best-effort; a failed poll simply retries on the next tick.
      }

      if (!cancelled) {
        timer = window.setTimeout(poll, intervalMs)
      }
    }

    void poll()

    return () => {
      cancelled = true
      controller.abort()

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [fetcher, active, intervalMs])

  return progress
}
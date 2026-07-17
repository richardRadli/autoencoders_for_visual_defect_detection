import { useCallback, useEffect, useRef, useState } from "react"

export type RunState =
  | "idle"
  | "running"
  | "done"
  | "stopped"
  | "error"

export type BlockingRun<P, R> = {
  state: RunState
  result: R | null
  error: string | null
  elapsedMs: number
  stopping: boolean
  start: (params: P) => Promise<void>
  stop: () => Promise<void>
  reset: () => void
}

/*
 * Drives the synchronous data_operations runs: the POST blocks until the job
 * finishes, so there is no task id and nothing to poll. The promise itself is
 * the progress signal.
 *
 * Elapsed time is measured here, on the client, because no endpoint reports it.
 * It starts on submit and freezes when the promise settles. A page reload loses
 * the timer state.
 */
export function useBlockingRun<P, R>(
  run: (params: P) => Promise<R>,
  requestStop: () => Promise<unknown>,
): BlockingRun<P, R> {
  const [state, setState] = useState<RunState>("idle")
  const [result, setResult] = useState<R | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [stopping, setStopping] = useState(false)

  const startedAt = useRef<number | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    if (state !== "running") {
      return
    }

    const id = window.setInterval(() => {
      if (startedAt.current !== null) {
        setElapsedMs(Date.now() - startedAt.current)
      }
    }, 1000)

    return () => window.clearInterval(id)
  }, [state])

  const start = useCallback(
    async (params: P) => {
      startedAt.current = Date.now()

      setState("running")
      setResult(null)
      setError(null)
      setElapsedMs(0)
      setStopping(false)

      try {
        const response = await run(params)

        if (!mounted.current) {
          return
        }

        setResult(response)
        setError(null)

        const responseStatus =
          typeof response === "object" &&
          response !== null &&
          "status" in response &&
          typeof response.status === "string"
            ? response.status.toLowerCase()
            : null

        setState(responseStatus === "stopped" ? "stopped" : "done")
      } catch (runError) {
        if (!mounted.current) {
          return
        }

        setError(
          runError instanceof Error
            ? runError.message
            : String(runError),
        )
        setState("error")
      } finally {
        if (mounted.current && startedAt.current !== null) {
          setElapsedMs(Date.now() - startedAt.current)
          setStopping(false)
        }
      }
    },
    [run],
  )

  const stop = useCallback(async () => {
    if (state !== "running" || stopping) {
      return
    }

    setError(null)
    setStopping(true)

    try {
      await requestStop()
    } catch (stopError) {
      if (!mounted.current) {
        return
      }

      setError(
        stopError instanceof Error
          ? stopError.message
          : String(stopError),
      )
      setStopping(false)
    }
  }, [requestStop, state, stopping])

  const reset = useCallback(() => {
    startedAt.current = null

    setState("idle")
    setResult(null)
    setError(null)
    setElapsedMs(0)
    setStopping(false)
  }, [])

  return {
    state,
    result,
    error,
    elapsedMs,
    stopping,
    start,
    stop,
    reset,
  }
}
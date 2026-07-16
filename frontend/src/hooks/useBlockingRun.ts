import { useCallback, useEffect, useRef, useState } from "react"

export type RunState = "idle" | "running" | "done" | "error"

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
 * finishes, so there is no task id and nothing to poll — the promise itself is
 * the progress signal.
 *
 * Elapsed time is measured here, on the client, because no endpoint reports it.
 * It starts on submit and freezes when the promise settles; a page reload loses
 * it.
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
        const res = await run(params)
        if (!mounted.current) {
          return
        }
        setResult(res)
        setState("done")
      } catch (e) {
        if (!mounted.current) {
          return
        }
        setError(e instanceof Error ? e.message : String(e))
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
    setStopping(true)
    try {
      await requestStop()
    } catch {
      // The run promise reports the real outcome; a failed stop signal alone
      // is not something the user can act on.
    }
  }, [requestStop])

  const reset = useCallback(() => {
    startedAt.current = null
    setState("idle")
    setResult(null)
    setError(null)
    setElapsedMs(0)
    setStopping(false)
  }, [])

  return { state, result, error, elapsedMs, stopping, start, stop, reset }
}
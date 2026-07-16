import { useCallback, useEffect, useRef, useState } from "react"

import { isAbortError } from "../api/client"
import type { QueuedTask, TaskState, TaskStatus } from "../api/types"

export type TaskRunState = "idle" | "running" | "done" | "error" | "stopped"

export type TaskRun<P, R> = {
  state: TaskRunState
  taskId: string | null
  taskState: TaskState | null
  result: R | null
  error: string | null
  elapsedMs: number
  stopping: boolean
  start: (params: P) => Promise<void>
  stop: () => Promise<void>
  reset: () => void
}


/*
 * Drives the Celery-backed train and test runs: the POST returns a task id
 * immediately and the real work is followed by polling the status endpoint.
 *
 * The task id lives in component state only — a page reload loses track of a
 * running task. Elapsed time is measured on the client and loses the same way.
 */
export function useTaskPolling<P, R>(
  run: (params: P) => Promise<QueuedTask>,
  getStatus: (taskId: string, signal?: AbortSignal) => Promise<TaskStatus<R>>,
  requestStop: (taskId: string) => Promise<unknown>,
  intervalMs = 2000,
): TaskRun<P, R> {
  const [state, setState] = useState<TaskRunState>("idle")
  const [taskId, setTaskId] = useState<string | null>(null)
  const [taskState, setTaskState] = useState<TaskState | null>(null)
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

  const freezeElapsed = useCallback(() => {
    if (startedAt.current !== null) {
      setElapsedMs(Date.now() - startedAt.current)
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

  useEffect(() => {
    if (state !== "running" || !taskId) {
      return
    }

    let cancelled = false
    let timer: number | undefined
    const controller = new AbortController()

    const settle = (next: TaskRunState) => {
      freezeElapsed()
      setStopping(false)
      setState(next)
    }

    const poll = async () => {
      try {
        const status = await getStatus(taskId, controller.signal)
        if (cancelled) {
          return
        }
        setTaskState(status.status)

        if (status.status === "SUCCESS") {
          setResult(status.info as R)
          settle("done")
          return
        }
        if (status.status === "FAILURE") {
          setError(typeof status.info === "string" ? status.info : "Task failed")
          settle("error")
          return
        }
        if (status.status === "REVOKED") {
          settle("stopped")
          return
        }
      } catch (e) {
        if (cancelled || isAbortError(e)) {
          return
        }
        setError(e instanceof Error ? e.message : String(e))
        settle("error")
        return
      }

      if (!cancelled) {
        timer = window.setTimeout(poll, intervalMs)
      }
    }

    poll()

    return () => {
      cancelled = true
      controller.abort()
      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [state, taskId, getStatus, intervalMs, freezeElapsed])

  const start = useCallback(
    async (params: P) => {
      startedAt.current = Date.now()
      setState("running")
      setTaskId(null)
      setTaskState(null)
      setResult(null)
      setError(null)
      setElapsedMs(0)
      setStopping(false)

      try {
        const queued = await run(params)
        if (!mounted.current) {
          return
        }
        setTaskId(queued.task_id)
        setTaskState(queued.status)
      } catch (e) {
        if (!mounted.current) {
          return
        }
        setError(e instanceof Error ? e.message : String(e))
        freezeElapsed()
        setState("error")
      }
    },
    [run, freezeElapsed],
  )

  const stop = useCallback(async () => {
    if (!taskId) {
      return
    }
    setStopping(true)
    try {
      await requestStop(taskId)
    } catch (e) {
      if (mounted.current) {
        setError(e instanceof Error ? e.message : String(e))
        setStopping(false)
      }
    }
  }, [taskId, requestStop])

  const reset = useCallback(() => {
    startedAt.current = null
    setState("idle")
    setTaskId(null)
    setTaskState(null)
    setResult(null)
    setError(null)
    setElapsedMs(0)
    setStopping(false)
  }, [])

  return {
    state,
    taskId,
    taskState,
    result,
    error,
    elapsedMs,
    stopping,
    start,
    stop,
    reset,
  }
}
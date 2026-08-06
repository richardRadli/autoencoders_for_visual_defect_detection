import { useCallback, useEffect, useRef, useState } from "react"

import { isAbortError } from "../api/client"
import type { QueuedTask, TaskProgress, TaskState, TaskStatus } from "../api/types"

export type TaskRunState = "idle" | "running" | "done" | "error" | "stopped"

export type TaskPollingOptions = {
  storageKey?: string
  intervalMs?: number
}

export type TaskRun<P, R> = {
  state: TaskRunState
  taskId: string | null
  taskState: TaskState | null
  progress: TaskProgress | null
  result: R | null
  error: string | null
  elapsedMs: number
  stopping: boolean
  submittedParams: P | null
  start: (params: P) => Promise<void>
  stop: () => Promise<void>
  reset: () => void
}

type PersistedTask<P, R> = {
  version: 1
  state: TaskRunState
  taskId: string | null
  taskState: TaskState | null
  result: R | null
  error: string | null
  elapsedMs: number
  startedAt: number | null
  submittedParams: P | null
}

const TASK_RUN_STATES: TaskRunState[] = [
  "idle",
  "running",
  "done",
  "error",
  "stopped",
]

const TASK_STATES: TaskState[] = [
  "QUEUED",
  "PENDING",
  "STARTED",
  "PROGRESS",
  "RETRY",
  "SUCCESS",
  "FAILURE",
  "REVOKED",
]

function createEmptyTask<P, R>(): PersistedTask<P, R> {
  return {
    version: 1,
    state: "idle",
    taskId: null,
    taskState: null,
    result: null,
    error: null,
    elapsedMs: 0,
    startedAt: null,
    submittedParams: null,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function isTaskRunState(value: unknown): value is TaskRunState {
  return (
    typeof value === "string" &&
    TASK_RUN_STATES.includes(value as TaskRunState)
  )
}

function isTaskState(value: unknown): value is TaskState {
  return (
    typeof value === "string" &&
    TASK_STATES.includes(value as TaskState)
  )
}

function removePersistedTask(storageKey?: string): void {
  if (!storageKey || typeof window === "undefined") {
    return
  }

  try {
    window.localStorage.removeItem(storageKey)
  } catch {
    // Persistence is optional. The task still works without localStorage.
  }
}

function savePersistedTask<P, R>(
  storageKey: string | undefined,
  task: PersistedTask<P, R>,
): void {
  if (!storageKey || typeof window === "undefined") {
    return
  }

  try {
    window.localStorage.setItem(storageKey, JSON.stringify(task))
  } catch {
    // Persistence is optional. The task still works without localStorage.
  }
}

function loadPersistedTask<P, R>(
  storageKey?: string,
): PersistedTask<P, R> {
  const emptyTask = createEmptyTask<P, R>()

  if (!storageKey || typeof window === "undefined") {
    return emptyTask
  }

  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) {
      return emptyTask
    }

    const parsed: unknown = JSON.parse(raw)

    if (
      !isRecord(parsed) ||
      parsed.version !== 1 ||
      !isTaskRunState(parsed.state)
    ) {
      removePersistedTask(storageKey)
      return emptyTask
    }

    const taskId =
      typeof parsed.taskId === "string" && parsed.taskId.length > 0
        ? parsed.taskId
        : null

    const taskState = isTaskState(parsed.taskState)
      ? parsed.taskState
      : null

    const startedAt =
      typeof parsed.startedAt === "number" &&
      Number.isFinite(parsed.startedAt)
        ? parsed.startedAt
        : null

    let elapsedMs =
      typeof parsed.elapsedMs === "number" &&
      Number.isFinite(parsed.elapsedMs) &&
      parsed.elapsedMs >= 0
        ? parsed.elapsedMs
        : 0

    if (parsed.state === "running") {
      if (taskId === null || startedAt === null) {
        removePersistedTask(storageKey)
        return emptyTask
      }

      elapsedMs = Math.max(0, Date.now() - startedAt)
    }

    return {
      version: 1,
      state: parsed.state,
      taskId,
      taskState,
      result: (parsed.result ?? null) as R | null,
      error: typeof parsed.error === "string" ? parsed.error : null,
      elapsedMs,
      startedAt,
      submittedParams: (parsed.submittedParams ?? null) as P | null,
    }
  } catch {
    removePersistedTask(storageKey)
    return emptyTask
  }
}

/*
 * Drives Celery-backed train and test runs. When storageKey is provided,
 * the task id, submitted parameters, result and elapsed time survive page
 * navigation and browser refreshes. Live PROGRESS meta is exposed as
 * `progress` (not persisted — the next poll repopulates it after a refresh).
 */
export function useTaskPolling<P, R>(
  run: (params: P) => Promise<QueuedTask>,
  getStatus: (
    taskId: string,
    signal?: AbortSignal,
  ) => Promise<TaskStatus<R>>,
  requestStop: (taskId: string) => Promise<unknown>,
  options: TaskPollingOptions | number = {},
): TaskRun<P, R> {
  const intervalMs =
    typeof options === "number"
      ? options
      : options.intervalMs ?? 2000

  const storageKey =
    typeof options === "number"
      ? undefined
      : options.storageKey

  const [initial] = useState<PersistedTask<P, R>>(() =>
    loadPersistedTask<P, R>(storageKey),
  )

  const [state, setState] = useState<TaskRunState>(initial.state)
  const [taskId, setTaskId] = useState<string | null>(initial.taskId)
  const [taskState, setTaskState] = useState<TaskState | null>(
    initial.taskState,
  )
  const [progress, setProgress] = useState<TaskProgress | null>(null)
  const [result, setResult] = useState<R | null>(initial.result)
  const [error, setError] = useState<string | null>(initial.error)
  const [elapsedMs, setElapsedMs] = useState(initial.elapsedMs)
  const [stopping, setStopping] = useState(false)
  const [submittedParams, setSubmittedParams] = useState<P | null>(
    initial.submittedParams,
  )

  const startedAt = useRef<number | null>(initial.startedAt)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    if (state === "idle") {
      removePersistedTask(storageKey)
      return
    }

    savePersistedTask<P, R>(storageKey, {
      version: 1,
      state,
      taskId,
      taskState,
      result,
      error,
      elapsedMs,
      startedAt: startedAt.current,
      submittedParams,
    })
  }, [
    storageKey,
    state,
    taskId,
    taskState,
    result,
    error,
    elapsedMs,
    submittedParams,
  ])

  const freezeElapsed = useCallback(() => {
    if (startedAt.current === null) {
      return
    }

    setElapsedMs(Math.max(0, Date.now() - startedAt.current))
  }, [])

  useEffect(() => {
    if (state !== "running") {
      return
    }

    const timer = window.setInterval(() => {
      if (startedAt.current !== null) {
        setElapsedMs(Math.max(0, Date.now() - startedAt.current))
      }
    }, 1000)

    return () => window.clearInterval(timer)
  }, [state])

  useEffect(() => {
    if (state !== "running" || !taskId) {
      return
    }

    let cancelled = false
    let timer: number | undefined
    const controller = new AbortController()

    const settle = (nextState: TaskRunState) => {
      freezeElapsed()
      setStopping(false)
      setState(nextState)
    }

    const poll = async () => {
      try {
        const status = await getStatus(taskId, controller.signal)

        if (cancelled) {
          return
        }

        setTaskState(status.status)

        if (status.status === "PROGRESS" && isRecord(status.info)) {
          setProgress(status.info as TaskProgress)
        }

        if (status.status === "SUCCESS") {
          setResult(status.info as R)
          setError(null)
          settle("done")
          return
        }

        if (status.status === "FAILURE") {
          setError(
            typeof status.info === "string"
              ? status.info
              : "Task failed",
          )
          settle("error")
          return
        }

        if (status.status === "REVOKED") {
          setError(null)
          settle("stopped")
          return
        }
      } catch (caughtError) {
        if (cancelled || isAbortError(caughtError)) {
          return
        }

        setError(
          caughtError instanceof Error
            ? caughtError.message
            : String(caughtError),
        )
        settle("error")
        return
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
  }, [
    state,
    taskId,
    getStatus,
    intervalMs,
    freezeElapsed,
  ])

  const start = useCallback(
    async (params: P) => {
      const startTime = Date.now()
      startedAt.current = startTime

      setState("running")
      setTaskId(null)
      setTaskState(null)
      setProgress(null)
      setResult(null)
      setError(null)
      setElapsedMs(0)
      setStopping(false)
      setSubmittedParams(params)

      try {
        const queued = await run(params)

        savePersistedTask<P, R>(storageKey, {
          version: 1,
          state: "running",
          taskId: queued.task_id,
          taskState: queued.status,
          result: null,
          error: null,
          elapsedMs: Math.max(0, Date.now() - startTime),
          startedAt: startTime,
          submittedParams: params,
        })

        if (!mounted.current) {
          return
        }

        setTaskId(queued.task_id)
        setTaskState(queued.status)
      } catch (caughtError) {
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : String(caughtError)

        const finalElapsed = Math.max(0, Date.now() - startTime)

        savePersistedTask<P, R>(storageKey, {
          version: 1,
          state: "error",
          taskId: null,
          taskState: null,
          result: null,
          error: message,
          elapsedMs: finalElapsed,
          startedAt: startTime,
          submittedParams: params,
        })

        if (!mounted.current) {
          return
        }

        setError(message)
        setElapsedMs(finalElapsed)
        setState("error")
      }
    },
    [run, storageKey],
  )

  const stop = useCallback(async () => {
    if (!taskId) {
      return
    }

    setStopping(true)

    try {
      await requestStop(taskId)
    } catch (caughtError) {
      if (mounted.current) {
        setError(
          caughtError instanceof Error
            ? caughtError.message
            : String(caughtError),
        )
        setStopping(false)
      }
    }
  }, [taskId, requestStop])

  const reset = useCallback(() => {
    removePersistedTask(storageKey)
    startedAt.current = null

    setState("idle")
    setTaskId(null)
    setTaskState(null)
    setProgress(null)
    setResult(null)
    setError(null)
    setElapsedMs(0)
    setStopping(false)
    setSubmittedParams(null)
  }, [storageKey])

  return {
    state,
    taskId,
    taskState,
    progress,
    result,
    error,
    elapsedMs,
    stopping,
    submittedParams,
    start,
    stop,
    reset,
  }
}
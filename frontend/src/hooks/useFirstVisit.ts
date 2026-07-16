import { useCallback, useEffect, useState } from "react"

const STORAGE_KEY = "defect-detection-guide-seen"
const PULSE_DURATION_MS = 4_000
const COACHMARK_DURATION_MS = 10_000
const FADE_DURATION_MS = 320

type CoachmarkPhase = "visible" | "fading" | "hidden"

type FirstVisitState = {
  phase: CoachmarkPhase
  pulseInfoButton: boolean
}

export type FirstVisitResult = {
  showCoachmark: boolean
  coachmarkFading: boolean
  pulseInfoButton: boolean
  dismissCoachmark: () => void
}

function hasSeenGuide(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "true"
  } catch {
    return false
  }
}

function rememberGuide(): void {
  try {
    localStorage.setItem(STORAGE_KEY, "true")
  } catch {
    // The coachmark still closes for the current session.
  }
}

export function useFirstVisit(): FirstVisitResult {
  const [state, setState] = useState<FirstVisitState>(() => {
    const firstVisit = !hasSeenGuide()

    return {
      phase: firstVisit ? "visible" : "hidden",
      pulseInfoButton: firstVisit,
    }
  })

  useEffect(() => {
    if (state.phase !== "visible") {
      return
    }

    const pulseTimer = window.setTimeout(() => {
      setState((current) => ({
        ...current,
        pulseInfoButton: false,
      }))
    }, PULSE_DURATION_MS)

    const fadeTimer = window.setTimeout(() => {
      rememberGuide()

      setState({
        phase: "fading",
        pulseInfoButton: false,
      })
    }, COACHMARK_DURATION_MS)

    return () => {
      window.clearTimeout(pulseTimer)
      window.clearTimeout(fadeTimer)
    }
  }, [state.phase])

  useEffect(() => {
    if (state.phase !== "fading") {
      return
    }

    const hideTimer = window.setTimeout(() => {
      setState({
        phase: "hidden",
        pulseInfoButton: false,
      })
    }, FADE_DURATION_MS)

    return () => window.clearTimeout(hideTimer)
  }, [state.phase])

  const dismissCoachmark = useCallback(() => {
    rememberGuide()

    setState({
      phase: "hidden",
      pulseInfoButton: false,
    })
  }, [])

  return {
    showCoachmark: state.phase !== "hidden",
    coachmarkFading: state.phase === "fading",
    pulseInfoButton: state.pulseInfoButton,
    dismissCoachmark,
  }
}
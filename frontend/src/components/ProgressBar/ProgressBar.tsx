import { useEffect, useRef, useState } from "react"

import styles from "./ProgressBar.module.css"

type ProgressBarProps = {
  current?: number
  total?: number
  phase?: string
  variant?: "accent" | "success"
  showCount?: boolean
  className?: string
}

const SUCCESS_HOLD_MS = 4000
const SUCCESS_FADE_MS = 350

export function ProgressBar({
  current,
  total,
  phase,
  variant = "accent",
  showCount = true,
  className,
}: ProgressBarProps) {
  const isSuccess = variant === "success"
  const previousVariant = useRef(variant)

  // A completed task loaded from storage should not replay the success bar.
  const [visible, setVisible] = useState(!isSuccess)
  const [fading, setFading] = useState(false)

  useEffect(() => {
    const wasSuccess = previousVariant.current === "success"
    previousVariant.current = variant

    let fadeTimer: number | undefined
    let hideTimer: number | undefined

    if (!isSuccess) {
      setVisible(true)
      setFading(false)
      return
    }

    // Show the completed bar only when a currently visible run changes
    // from in progress to success.
    if (!wasSuccess) {
      setVisible(true)
      setFading(false)

      fadeTimer = window.setTimeout(() => {
        setFading(true)
      }, SUCCESS_HOLD_MS)

      hideTimer = window.setTimeout(() => {
        setVisible(false)
      }, SUCCESS_HOLD_MS + SUCCESS_FADE_MS)
    }

    return () => {
      if (fadeTimer !== undefined) {
        window.clearTimeout(fadeTimer)
      }

      if (hideTimer !== undefined) {
        window.clearTimeout(hideTimer)
      }
    }
  }, [isSuccess, variant])

  if (!visible) {
    return null
  }

  const value = typeof current === "number" && current > 0 ? current : 0
  const hasTotal = typeof total === "number" && total > 0

  const percent = hasTotal
    ? Math.min(100, Math.max(0, Math.round((value / total) * 100)))
    : 0

  const count =
    hasTotal && showCount ? `${Math.min(value, total)}/${total}` : null

  const subLabel = [phase, count].filter(Boolean).join(" · ") || null
  const fillClass = isSuccess ? styles.fillSuccess : styles.fillAccent

  const classes = [styles.wrap, fading ? styles.fading : null, className]
    .filter(Boolean)
    .join(" ")

  return (
    <div className={classes}>
      <span className={styles.percent}>{percent}%</span>

      <div
        className={styles.track}
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={`${styles.fill} ${fillClass}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      {subLabel ? <span className={styles.sub}>{subLabel}</span> : null}
    </div>
  )
}
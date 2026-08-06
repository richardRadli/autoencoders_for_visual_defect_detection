import styles from "./ProgressBar.module.css"

type ProgressBarProps = {
  current?: number
  total?: number
  phase?: string
  variant?: "accent" | "success"
  className?: string
}

/*
 * Shared, presentational progress bar: a thin track with a filled portion and
 * a "42% · phase · current/total" label. The caller decides what to pass, so
 * the same bar serves training (epochs), testing (processed items) and the
 * data_operations runs (processed images). accent = in progress (blue),
 * success = finished / early stopped (green). When no positive total is known
 * yet (e.g. the setup phase), the bar shows 0% and, if given, just the phase.
 */
export function ProgressBar({
  current,
  total,
  phase,
  variant = "accent",
  className,
}: ProgressBarProps) {
  const value = typeof current === "number" && current > 0 ? current : 0

  const percent =
    total && total > 0
      ? Math.min(100, Math.max(0, Math.round((value / total) * 100)))
      : 0

  const label =
    total && total > 0
      ? [`${percent}%`, phase, `${Math.min(value, total)}/${total}`]
          .filter(Boolean)
          .join(" · ")
      : phase ?? null

  const fillClass =
    variant === "success" ? styles.fillSuccess : styles.fillAccent

  const classes = [styles.wrap, className].filter(Boolean).join(" ")

  return (
    <div className={classes}>
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

      {label ? <span className={styles.label}>{label}</span> : null}
    </div>
  )
}
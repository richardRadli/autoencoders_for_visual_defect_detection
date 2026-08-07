import styles from "./ProgressBar.module.css"

type ProgressBarProps = {
  current?: number
  total?: number
  phase?: string
  variant?: "accent" | "success"
  showCount?: boolean
  className?: string
}

/*
 * Shared, presentational progress bar. A big centered percentage sits above a
 * thin track; a small phase (and, when showCount is set, a current/total count)
 * sits below. The caller supplies the numbers and labels, so the same bar
 * serves training (epochs), testing (normalized percent), and the
 * data_operations / tuning runs. accent = in progress (blue), success =
 * finished (green). On success the bar "settles": it holds full green for a few
 * seconds, then the percent and details fade out via CSS, leaving a faint green
 * track. Remounting (returning to the page) replays that fade. With no positive
 * total the bar shows 0%.
 */
export function ProgressBar({
  current,
  total,
  phase,
  variant = "accent",
  showCount = true,
  className,
}: ProgressBarProps) {
  const value = typeof current === "number" && current > 0 ? current : 0
  const hasTotal = typeof total === "number" && total > 0

  const percent = hasTotal
    ? Math.min(100, Math.max(0, Math.round((value / total) * 100)))
    : 0

  const count =
    hasTotal && showCount ? `${Math.min(value, total)}/${total}` : null

  const subLabel = [phase, count].filter(Boolean).join(" · ") || null

  const settled = variant === "success"
  const fillClass = settled ? styles.fillSuccess : styles.fillAccent

  const classes = [styles.wrap, settled ? styles.settled : null, className]
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
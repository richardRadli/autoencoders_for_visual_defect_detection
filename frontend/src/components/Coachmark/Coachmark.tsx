import { X } from "lucide-react"
import type { ReactNode } from "react"

import { IconButton } from "../IconButton/IconButton"
import styles from "./Coachmark.module.css"

type CoachmarkProps = {
  onDismiss: () => void
  fading?: boolean
  className?: string
  children: ReactNode
}

export function Coachmark({
  onDismiss,
  fading = false,
  className,
  children,
}: CoachmarkProps) {
  const classes = [
    styles.coachmark,
    fading ? styles.fading : null,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <div className={classes} role="status" aria-live="polite">
      <span className={styles.arrow} aria-hidden="true" />
      <p className={styles.message}>{children}</p>
      <IconButton
        icon={X}
        label="Dismiss tip"
        size="small"
        className={styles.dismiss}
        onClick={onDismiss}
      />
    </div>
  )
}
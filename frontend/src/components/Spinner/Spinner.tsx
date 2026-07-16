import { LoaderCircle } from "lucide-react"

import styles from "./Spinner.module.css"

export type SpinnerSize = "small" | "medium" | "large"

type SpinnerProps = {
  size?: SpinnerSize
  label?: string
  className?: string
}

export function Spinner({
  size = "medium",
  label,
  className,
}: SpinnerProps ) {
  const classes = [
    styles.spinner,
    styles[size],
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <span
      className={classes}
      role={label ? "status" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <LoaderCircle className={styles.icon} aria-hidden="true" />
    </span>
  )
}
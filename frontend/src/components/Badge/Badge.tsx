import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"

import styles from "./Badge.module.css"

export type BadgeVariant = "neutral" | "accent" | "success" | "warning" | "danger"

type BadgeProps = {
  variant?: BadgeVariant
  icon?: LucideIcon
  children: ReactNode
  className?: string
}

export function Badge({
  variant = "neutral",
  icon: Icon,
  children,
  className,
}: BadgeProps) {
  const classes = [
    styles.badge,
    styles[variant],
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <span className={classes}>
      {Icon ? <Icon className={styles.icon} aria-hidden="true" /> : null}
      <span className={styles.label}>{children}</span>
    </span>
  )
}
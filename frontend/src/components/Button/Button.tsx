import type { LucideIcon } from "lucide-react"
import type { ButtonHTMLAttributes, ReactNode } from "react"

import { Spinner } from "../Spinner/Spinner"
import styles from "./Button.module.css"

export type ButtonVariant = "primary" | "secondary" | "danger"

type ButtonProps = {
  variant?: ButtonVariant
  icon?: LucideIcon
  iconPosition?: "left" | "right"
  loading?: boolean
  children: ReactNode
} & ButtonHTMLAttributes<HTMLButtonElement>

export function Button({
  variant = "secondary",
  icon: Icon,
  iconPosition = "left",
  loading = false,
  children,
  className,
  disabled,
  type = "button",
  ...props
}: ButtonProps) {
  const classes = [
    styles.button,
    styles[variant],
    className,
  ]
    .filter(Boolean)
    .join(" ")

  let glyph: ReactNode = null
  if (loading) {
    glyph = <Spinner size="small" />
  } else if (Icon) {
    glyph = <Icon className={styles.icon} aria-hidden="true" />
  }

  return (
    <button
      type={type}
      className={classes}
      disabled={disabled || loading}
      {...props}
    >
      {iconPosition === "left" ? glyph : null}
      <span className={styles.label}>{children}</span>
      {iconPosition === "right" ? glyph : null}
    </button>
  )
}
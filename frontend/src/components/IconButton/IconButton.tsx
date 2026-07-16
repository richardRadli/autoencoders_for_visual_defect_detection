import type { LucideIcon } from "lucide-react"
import type { ButtonHTMLAttributes } from "react"

import styles from "./IconButton.module.css"

export type IconButtonVariant = "ghost" | "accent"
export type IconButtonSize = "small" | "medium"

type IconButtonProps = {
  icon: LucideIcon
  label: string
  variant?: IconButtonVariant
  size?: IconButtonSize
} & ButtonHTMLAttributes<HTMLButtonElement>

export function IconButton({
  icon: Icon,
  label,
  variant = "ghost",
  size = "medium",
  className,
  type = "button",
  ...props
}: IconButtonProps) {
  const classes = [
    styles.iconButton,
    styles[variant],
    styles[size],
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <button type={type} className={classes} aria-label={label} {...props}>
      <Icon className={styles.icon} aria-hidden="true" />
    </button>
  )
}
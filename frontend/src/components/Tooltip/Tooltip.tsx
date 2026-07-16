import { useId, useState } from "react"
import type { KeyboardEvent, ReactNode } from "react"

import styles from "./Tooltip.module.css"

export type TooltipPlacement = "top" | "bottom" | "left" | "right"

type TooltipProps = {
  content: ReactNode
  children: ReactNode
  placement?: TooltipPlacement
  className?: string
}

export function Tooltip({
  content,
  children,
  placement = "bottom",
  className,
}: TooltipProps) {
  const [open, setOpen] = useState(false)
  const tooltipId = useId()

  const classes = [
    styles.wrapper,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  const handleKeyDown = (event: KeyboardEvent<HTMLSpanElement>) => {
    if (event.key === "Escape") {
      setOpen(false)
    }
  }

  return (
    <span
      className={classes}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onKeyDown={handleKeyDown}
      aria-describedby={open ? tooltipId : undefined}
    >
      {children}
      {open ? (
        <span
          id={tooltipId}
          role="tooltip"
          className={[styles.tooltip, styles[placement]].join(" ")}
        >
          {content}
        </span>
      ) : null}
    </span>
  )
}
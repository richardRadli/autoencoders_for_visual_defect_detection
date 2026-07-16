import { Info } from "lucide-react"

import { IconButton } from "../IconButton/IconButton"
import styles from "./InfoButton.module.css"

type InfoButtonProps = {
  onClick: () => void
  pulse?: boolean
  label?: string
  className?: string
}

export function InfoButton({
  onClick,
  pulse = false,
  label = "Open the guide",
  className,
}: InfoButtonProps) {
  const classes = [
    styles.infoButton,
    pulse ? styles.pulse : null,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <IconButton
      icon={Info}
      label={label}
      variant="accent"
      onClick={onClick}
      className={classes}
    />
  )
}
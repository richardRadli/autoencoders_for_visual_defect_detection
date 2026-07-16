import { Play, Square } from "lucide-react"

import { Button } from "../Button/Button"
import styles from "./RunControls.module.css"

type RunControlsProps = {
  running: boolean
  stopping?: boolean
  onStart: () => void
  onStop: () => void
  startLabel: string
  disabled?: boolean
  className?: string
}

export function RunControls({
  running,
  stopping = false,
  onStart,
  onStop,
  startLabel,
  disabled = false,
  className,
}: RunControlsProps) {
  const classes = [
    styles.controls,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <div className={classes}>
      <Button
        className={styles.start}
        variant="primary"
        icon={Play}
        onClick={onStart}
        loading={running}
        disabled={disabled}
      >
        {running ? "Running…" : startLabel}
      </Button>

      <Button
        className={styles.stop}
        variant="danger"
        icon={Square}
        onClick={onStop}
        loading={stopping}
        disabled={!running}
      >
        {stopping ? "Stopping…" : "Stop"}
      </Button>
    </div>
  )
}
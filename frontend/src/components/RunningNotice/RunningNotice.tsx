import { X } from "lucide-react"
import { useState } from "react"

import { Button } from "../Button/Button"
import styles from "./RunningNotice.module.css"

type RunningNoticeProps = {
  running: boolean
  label: string
}

export function RunningNotice({ running, label }: RunningNoticeProps) {
  // Capture whether it was already running when this mounted — i.e. the user
  // returned to the page mid-run. A fresh start (mounts idle) never triggers it.
  const [wasRunningOnMount] = useState(running)
  const [dismissed, setDismissed] = useState(false)

  if (!wasRunningOnMount || !running || dismissed) {
    return null
  }

  return (
    <div className={styles.overlay}>
      <div
        className={styles.box}
        role="dialog"
        aria-modal="true"
        aria-label={`${label} status`}
      >
        <button
          type="button"
          className={styles.close}
          aria-label="Close"
          onClick={() => setDismissed(true)}
        >
          <X className={styles.closeIcon} aria-hidden="true" />
        </button>

        <p className={styles.text}>{label} is still running</p>
        <p className={styles.subtext}>
          The task continued while you were away.
        </p>

        <div className={styles.actions}>
          <Button variant="primary" onClick={() => setDismissed(true)}>
            OK
          </Button>
        </div>
      </div>
    </div>
  )
}
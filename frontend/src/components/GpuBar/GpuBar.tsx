import { Cpu } from "lucide-react"

import { useDeviceStatus } from "../../hooks/useDeviceStatus"
import { Spinner } from "../Spinner/Spinner"
import styles from "./GpuBar.module.css"

type GpuBarProps = {
  intervalMs?: number
  className?: string
}

export function GpuBar({ intervalMs, className }: GpuBarProps) {
  const { data, loading, error } = useDeviceStatus(intervalMs)

  const classes = [
    styles.gpuBar,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  if (loading && !data) {
    return (
      <div className={classes}>
        <Cpu className={styles.icon} aria-hidden="true" />
        <span className={styles.label}>Checking device</span>
        <Spinner size="small" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className={classes}>
        <Cpu className={styles.icon} aria-hidden="true" />
        <span className={styles.label}>Device status unavailable</span>
      </div>
    )
  }

  if (!data.cuda_available) {
    return (
      <div className={classes}>
        <Cpu className={styles.icon} aria-hidden="true" />
        <span className={styles.label}>CPU</span>
        <span className={styles.detail}>{data.message}</span>
      </div>
    )
  }

  // `usage` is used VRAM in GB, not a percentage — the backend sends
  // (total - free), so the ratio has to be computed here.
  const used = data.usage
  const total = data.total_vram_gb
  const ratio = total > 0 ? Math.min(Math.max(used / total, 0), 1) : 0
  const percent = Math.round(ratio * 100)

  let level = styles.levelNormal
  if (percent >= 90) {
    level = styles.levelDanger
  } else if (percent >= 75) {
    level = styles.levelWarning
  }

  return (
    <div className={classes}>
      <Cpu className={styles.icon} aria-hidden="true" />
      <span className={styles.label} title={data.device_name}>
        {data.device_name}
      </span>

      <div
        className={styles.track}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-label={`VRAM usage: ${percent}%`}
      >
        <div
          className={`${styles.fill} ${level}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      <span className={styles.detail}>
        {used.toFixed(1)} / {total.toFixed(1)} GB
      </span>
    </div>
  )
}
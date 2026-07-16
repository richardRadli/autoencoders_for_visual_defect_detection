import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"

import { Spinner } from "../Spinner/Spinner"
import styles from "./StatusGrid.module.css"

export type StatusItem = {
  label: string
  value?: ReactNode
  icon?: LucideIcon
  title?: string
}

type StatusGridProps = {
  items: StatusItem[]
  loading?: boolean
  className?: string
}

function renderValue(value: ReactNode, loading: boolean): ReactNode {
  if (loading) {
    return <Spinner size="small" label="Loading" />
  }

  if (value === null || value === undefined || value === "") {
    return <span className={styles.placeholder}>—</span>
  }

  return value
}

export function StatusGrid({
  items,
  loading = false,
  className,
}: StatusGridProps) {
  const classes = [
    styles.grid,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <div className={classes}>
      {items.map(({ label, value, icon: Icon, title }) => (
        <div className={styles.tile} key={label}>
          <div className={styles.head}>
            {Icon ? <Icon className={styles.icon} aria-hidden="true" /> : null}
            <span className={styles.label}>{label}</span>
          </div>

          <div className={styles.value} title={title}>
            {renderValue(value, loading)}
          </div>
        </div>
      ))}
    </div>
  )
}
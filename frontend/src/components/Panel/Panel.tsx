import type { ReactNode } from "react"

import styles from "./Panel.module.css"

type PanelProps = {
  title?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}

export function Panel({
  title,
  actions,
  children,
  className,
  bodyClassName,
}: PanelProps) {
  const classes = [
    styles.panel,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  const bodyClasses = [
    styles.body,
    bodyClassName,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <section className={classes}>
      {title || actions ? (
        <header className={styles.header}>
          {title ? <h2 className={styles.title}>{title}</h2> : null}
          {actions ? <div className={styles.actions}>{actions}</div> : null}
        </header>
      ) : null}
      <div className={bodyClasses}>{children}</div>
    </section>
  )
}
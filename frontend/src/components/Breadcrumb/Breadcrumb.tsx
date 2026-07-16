import { House } from "lucide-react"
import { Link } from "react-router-dom"

import {
  MAIN_MENU_LABEL,
  MAIN_MENU_PATH,
  getService,
  getStep,
} from "../../config/workflow"
import type { StepId } from "../../config/workflow"
import styles from "./Breadcrumb.module.css"

type BreadcrumbProps = {
  step: StepId
  className?: string
}

export function Breadcrumb({ step, className }: BreadcrumbProps) {
  const current = getStep(step)
  const service = getService(step)

  const classes = [
    styles.breadcrumb,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <nav className={classes} aria-label="Breadcrumb">
      <ol className={styles.list}>
        <li className={styles.item}>
          <Link className={styles.link} to={MAIN_MENU_PATH}>
            <House className={styles.icon} aria-hidden="true" />
            <span>{MAIN_MENU_LABEL}</span>
          </Link>
        </li>

        <li className={styles.item} aria-hidden="true">
          <span className={styles.separator}>/</span>
        </li>

        <li className={styles.item}>
          <span className={styles.service}>{service.label}</span>
        </li>

        <li className={styles.item} aria-hidden="true">
          <span className={styles.separator}>/</span>
        </li>

        <li className={styles.item}>
          <span className={styles.current} aria-current="page">
            {current.label}
          </span>
        </li>
      </ol>
    </nav>
  )
}
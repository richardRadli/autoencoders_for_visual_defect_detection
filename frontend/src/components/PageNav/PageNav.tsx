import { ChevronLeft, ChevronRight } from "lucide-react"
import { Link } from "react-router-dom"

import {
  WORKFLOW_STEPS,
  getNextStep,
  getPreviousStep,
  getStepNumber,
} from "../../config/workflow"
import type { StepId } from "../../config/workflow"
import styles from "./PageNav.module.css"

type PageNavProps = {
  step: StepId
  className?: string
}

export function PageNav({ step, className }: PageNavProps) {
  const previous = getPreviousStep(step)
  const next = getNextStep(step)

  const classes = [
    styles.pageNav,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <nav className={classes} aria-label="Workflow step navigation">
      <div className={styles.side}>
        {previous ? (
          <Link
            className={styles.link}
            to={previous.path}
            aria-label={`Previous step: ${previous.label}`}
          >
            <ChevronLeft className={styles.icon} aria-hidden="true" />
            <span className={styles.label}>{previous.label}</span>
          </Link>
        ) : null}
      </div>

      <span className={styles.counter}>
        Step {getStepNumber(step)} of {WORKFLOW_STEPS.length}
      </span>

      <div className={`${styles.side} ${styles.sideEnd}`}>
        {next ? (
          <Link
            className={styles.link}
            to={next.path}
            aria-label={`Next step: ${next.label}`}
          >
            <span className={styles.label}>{next.label}</span>
            <ChevronRight className={styles.icon} aria-hidden="true" />
          </Link>
        ) : null}
      </div>
    </nav>
  )
}
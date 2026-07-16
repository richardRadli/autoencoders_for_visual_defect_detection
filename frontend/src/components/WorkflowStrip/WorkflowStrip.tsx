import { Check } from "lucide-react"
import { Link } from "react-router-dom"

import { WORKFLOW_STEPS } from "../../config/workflow"
import type { StepId } from "../../config/workflow"
import { Spinner } from "../Spinner/Spinner"
import styles from "./WorkflowStrip.module.css"

export type StepState = "complete" | "ready" | "running" | "missing"

export type StepInfo = {
  state: StepState
  detail: string
}

const STATE_LABEL: Record<StepState, string> = {
  complete: "Complete",
  ready: "Ready",
  running: "Running",
  missing: "Not ready",
}

type WorkflowStripProps = {
  steps: Record<StepId, StepInfo>
  current?: StepId
  className?: string
}

export function WorkflowStrip({
  steps,
  current,
  className,
}: WorkflowStripProps) {
  const classes = [
    styles.workflowStrip,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <nav className={classes} aria-label="Workflow">
      <ol className={styles.list}>
        {WORKFLOW_STEPS.map((step, index) => {
          const info = steps[step.id]
          const isCurrent = step.id === current

          const itemClasses = [
            styles.item,
            styles[info.state],
            isCurrent ? styles.current : null,
          ]
            .filter(Boolean)
            .join(" ")

          return (
            <li className={itemClasses} key={step.id}>
              <Link
                className={styles.card}
                to={step.path}
                aria-current={isCurrent ? "step" : undefined}
                aria-label={`${index + 1}. ${step.label}. ${STATE_LABEL[info.state]}. ${info.detail}`}
              >
                <span className={styles.head}>
                  <span className={styles.marker} aria-hidden="true">
                    {info.state === "complete" ? (
                      <Check className={styles.check} />
                    ) : (
                      index + 1
                    )}
                  </span>
                  <span className={styles.label}>{step.label}</span>
                </span>

                <span className={styles.status}>
                  {info.state === "running" ? (
                    <Spinner size="small" />
                  ) : (
                    <span className={styles.dot} aria-hidden="true" />
                  )}
                  <span className={styles.statusLabel}>
                    {STATE_LABEL[info.state]}
                  </span>
                </span>

                <span className={styles.detail}>{info.detail}</span>
              </Link>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
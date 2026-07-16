import { X } from "lucide-react"
import { useEffect, useRef } from "react"
import { createPortal } from "react-dom"

import { SERVICES, getServiceSteps, getStepNumber } from "../../config/workflow"
import type { StepId } from "../../config/workflow"
import { IconButton } from "../IconButton/IconButton"
import styles from "./GuideModal.module.css"

const TITLE_ID = "guide-modal-title"

/* One explanation per workflow step. Typed against StepId, so a new step in
   workflow.ts cannot ship without its guide text. */
const STEP_HELP: Record<StepId, string> = {
  augmentation:
    "Turns a handful of good images into thousands by rotating and flipping them, then crops them to the size the network expects.",
  "draw-rectangles":
    "Covers parts of the augmented images with grey rectangles. The network learns to rebuild what is hidden — that is what makes it notice defects later.",
  training:
    "Trains an autoencoder on the prepared images. It only ever sees good parts, so it gets good at reconstructing good parts, and bad at reconstructing defects.",
  testing:
    "Runs the trained network over the test images and scores them. Where the reconstruction differs from the original, there is probably a defect.",
}

type GuideModalProps = {
  open: boolean
  onClose: () => void
}

export function GuideModal({ open, onClose }: GuideModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [open, onClose])

  useEffect(() => {
    if (!open) {
      return
    }

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"

    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [open])

  // Focus moves into the dialog on open and back to the trigger on close,
  // so keyboard users are not dropped at the top of the page.
  useEffect(() => {
    if (!open) {
      return
    }

    const previouslyFocused = document.activeElement as HTMLElement | null
    dialogRef.current?.focus()

    return () => previouslyFocused?.focus()
  }, [open])

  if (!open) {
    return null
  }

  return createPortal(
    <div
      className={styles.overlay}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose()
        }
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={TITLE_ID}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div className={styles.heading}>
            <h2 className={styles.title} id={TITLE_ID}>
              How this works
            </h2>
            <p className={styles.subtitle}>
              Run the steps in order — each one prepares what the next needs.
            </p>
          </div>

          <IconButton icon={X} label="Close guide" onClick={onClose} />
        </header>

        <div className={styles.body}>
          {Object.values(SERVICES).map((service) => (
            <section className={styles.service} key={service.id}>
              <h3 className={styles.serviceTitle}>{service.label}</h3>
              <p className={styles.serviceDescription}>{service.description}</p>

              <ol className={styles.steps}>
                {getServiceSteps(service.id).map((step) => (
                  <li className={styles.step} key={step.id}>
                    <span className={styles.stepNumber} aria-hidden="true">
                      {getStepNumber(step.id)}
                    </span>
                    <div className={styles.stepText}>
                      <p className={styles.stepLabel}>{step.label}</p>
                      <p className={styles.stepHelp}>{STEP_HELP[step.id]}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  )
}
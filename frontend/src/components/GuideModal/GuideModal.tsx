import { X } from "lucide-react"
import { useEffect, useRef } from "react"
import { createPortal } from "react-dom"

import { SERVICES, getServiceSteps, getStepNumber } from "../../config/workflow"
import type { StepId } from "../../config/workflow"
import { IconButton } from "../IconButton/IconButton"
import styles from "./GuideModal.module.css"

const TITLE_ID = "guide-modal-title"

const STEP_HELP: Record<StepId, string> = {
  augmentation:
    'Creates changed copies of the images in the "good" folder by rotating, flipping, and cropping them. The new images are saved in "aug" and used by Training.',
  "draw-rectangles":
    'Covers parts of the images in "aug" with gray rectangles and saves the results in "noise". This is needed only when ae_type is set to denoising. The number of "aug" and "noise" images must match.',
  training:
    'Uses the images in "aug" to train the selected model. Denoising models also use the matching images in "noise". The best weights are saved for Testing.',
  testing:
    "Loads the weights you select and checks the images in the test folder. It marks areas that may contain defects and saves result images and scores.",
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
              This app detects visual defects in images. It learns from images
              in the "good" folder, then looks for possible defects in test
              images. Follow the steps below in order.
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

                {service.id === "defect_detection" ? (
                  <li className={styles.step}>
                    <span className={styles.stepNumber} aria-hidden="true">
                      *
                    </span>

                    <div className={styles.stepText}>
                      <p className={styles.stepLabel}>
                        Parameter tuning (Optuna, optional)
                      </p>

                      <p className={styles.stepHelp}>
                        Tries different values for five Training settings and
                        shows which combination worked best. It does not save
                        weights or a finished model. Copy the returned values
                        to Training if you want to use them.
                      </p>
                    </div>
                  </li>
                ) : null}
              </ol>
            </section>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  )
}
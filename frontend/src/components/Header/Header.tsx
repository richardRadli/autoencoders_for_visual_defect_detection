import { Crosshair, Moon, Sun } from "lucide-react"
import { useState } from "react"

import { SERVICES } from "../../config/workflow"
import { useFirstVisit } from "../../hooks/useFirstVisit"
import { useServiceHealth } from "../../hooks/useServiceHealth"
import type { HealthState } from "../../hooks/useServiceHealth"
import { useTheme } from "../../theme/ThemeContext"
import { Coachmark } from "../Coachmark/Coachmark"
import { GuideModal } from "../GuideModal/GuideModal"
import { IconButton } from "../IconButton/IconButton"
import { InfoButton } from "../InfoButton/InfoButton"
import styles from "./Header.module.css"

const HEALTH_LABEL: Record<HealthState, string> = {
  checking: "checking…",
  online: "online",
  offline: "offline",
}

export function Header() {
  const { theme, toggleTheme } = useTheme()
  const { health } = useServiceHealth()
  const {
    showCoachmark,
    coachmarkFading,
    pulseInfoButton,
    dismissCoachmark,
  } = useFirstVisit()

  const [guideOpen, setGuideOpen] = useState(false)

  function handleGuideOpen() {
    dismissCoachmark()
    setGuideOpen(true)
  }

  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <Crosshair className={styles.logo} aria-hidden="true" />
        <span className={styles.title}>Defect detection framework</span>
      </div>

      <div className={styles.actions}>
        <div className={styles.services}>
          {Object.values(SERVICES).map((service) => {
            const state = health[service.id]
            const description = `${service.id}: ${HEALTH_LABEL[state]}`

            return (
              <span
                key={service.id}
                className={styles.service}
                title={description}
              >
                <span
                  className={[styles.dot, styles[state]].join(" ")}
                  role="status"
                  aria-label={description}
                />
                <span className={styles.serviceLabel}>{service.id}</span>
              </span>
            )
          })}
        </div>

        <div className={styles.guideAnchor}>
          <InfoButton onClick={handleGuideOpen} pulse={pulseInfoButton} />

          {showCoachmark ? (
            <Coachmark
              fading={coachmarkFading}
              onDismiss={dismissCoachmark}
            >
              New here? Open the guide to see how it works.
            </Coachmark>
          ) : null}
        </div>

        <IconButton
          icon={theme === "light" ? Moon : Sun}
          label={
            theme === "light"
              ? "Switch to dark theme"
              : "Switch to light theme"
          }
          onClick={toggleTheme}
        />
      </div>

      <GuideModal open={guideOpen} onClose={() => setGuideOpen(false)} />
    </header>
  )
}
import { X } from "lucide-react"
import { useEffect } from "react"
import { createPortal } from "react-dom"

import { IconButton } from "../IconButton/IconButton"
import styles from "./Lightbox.module.css"

type LightboxProps = {
  src: string
  alt: string
  onClose: () => void
}

export function Lightbox({ src, alt, onClose }: LightboxProps) {
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow

    document.body.style.overflow = "hidden"

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose()
      }
    }

    document.addEventListener("keydown", handleKeyDown)

    return () => {
      document.removeEventListener("keydown", handleKeyDown)
      document.body.style.overflow = previousOverflow
      previouslyFocused?.focus()
    }
  }, [onClose])

  return createPortal(
    <div
      className={styles.backdrop}
      role="dialog"
      aria-modal="true"
      aria-label={alt}
      onClick={onClose}
    >
      <div
        className={styles.content}
        onClick={(event) => event.stopPropagation()}
      >
        <IconButton
          autoFocus
          icon={X}
          label="Close preview"
          size="small"
          className={styles.close}
          onClick={onClose}
        />

        <img className={styles.image} src={src} alt={alt} />
      </div>
    </div>,
    document.body,
  )
}
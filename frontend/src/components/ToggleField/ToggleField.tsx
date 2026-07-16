import { Info } from "lucide-react"
import { useId } from "react"
import type { ReactNode } from "react"

import { Tooltip } from "../Tooltip/Tooltip"
import styles from "./ToggleField.module.css"

type ToggleFieldProps = {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  tooltip?: ReactNode
  hint?: ReactNode
  disabled?: boolean
  className?: string
}

export function ToggleField({
  label,
  checked,
  onChange,
  tooltip,
  hint,
  disabled = false,
  className,
}: ToggleFieldProps) {
  const fieldId = useId()

  const classes = [
    styles.field,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <div className={classes}>
      <div className={styles.row}>
        <input
          id={fieldId}
          type="checkbox"
          role="switch"
          className={styles.input}
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />

        <label className={styles.label} htmlFor={fieldId}>
          {label}
        </label>

        {tooltip ? (
          <Tooltip content={tooltip}>
            <button
              type="button"
              className={styles.info}
              aria-label={`About ${label}`}
            >
              <Info className={styles.infoIcon} aria-hidden="true" />
            </button>
          </Tooltip>
        ) : null}
      </div>

      {hint ? <p className={styles.hint}>{hint}</p> : null}
    </div>
  )
}
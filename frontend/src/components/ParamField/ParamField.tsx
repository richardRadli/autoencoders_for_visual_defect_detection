import { Info } from "lucide-react"
import { cloneElement, useId } from "react"
import type { ReactElement, ReactNode } from "react"

import { Tooltip } from "../Tooltip/Tooltip"
import styles from "./ParamField.module.css"

type ParamFieldChildProps = {
  id?: string
}

type ParamFieldProps = {
  label: string
  tooltip?: ReactNode
  hint?: ReactNode
  children: ReactElement<ParamFieldChildProps>
  className?: string
}

export function ParamField({
  label,
  tooltip,
  hint,
  children,
  className,
}: ParamFieldProps) {
  const generatedId = useId()
  const fieldId = children.props.id ?? generatedId
  const field = children.props.id
    ? children
    : cloneElement(children, { id: fieldId })

  const classes = [
    styles.field,
    className,
  ]
    .filter(Boolean)
    .join(" ")

  return (
    <div className={classes}>
      <div className={styles.labelRow}>
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

      {field}

      {hint ? <p className={styles.hint}>{hint}</p> : null}
    </div>
  )
}
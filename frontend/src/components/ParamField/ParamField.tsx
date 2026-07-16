import { useState } from "react"
import type { ReactNode } from "react"

export function ParamField({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  const [show, setShow] = useState(false)

  return (
    <div>
      <label className="field-label" style={{ display: "flex", alignItems: "center", gap: 5 }}>
        {label}
        {hint && (
          <span
            style={{ position: "relative", display: "inline-flex", cursor: "help" }}
            onMouseEnter={() => setShow(true)}
            onMouseLeave={() => setShow(false)}
          >
            <span
              style={{
                width: 15,
                height: 15,
                borderRadius: "50%",
                border: "0.5px solid var(--border-strong)",
                color: "var(--text-muted)",
                fontSize: 10,
                fontStyle: "italic",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                lineHeight: 1,
              }}
            >
              i
            </span>
            {show && (
              <span
                style={{
                  position: "absolute",
                  top: "calc(100% + 6px)",
                  left: 0,
                  width: 200,
                  background: "var(--surface-2)",
                  border: "0.5px solid var(--border-strong)",
                  borderRadius: 8,
                  padding: "8px 10px",
                  fontSize: 12,
                  fontWeight: 400,
                  color: "var(--text-secondary)",
                  lineHeight: 1.4,
                  zIndex: 10,
                  boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
                }}
              >
                {hint}
              </span>
            )}
          </span>
        )}
      </label>
      {children}
    </div>
  )
}
import { Check, Copy, Eye, EyeOff } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import type { ReactNode } from "react"

import { Button } from "../Button/Button"
import { Panel } from "../Panel/Panel"
import styles from "./JsonPanel.module.css"

type JsonPanelProps = {
  title?: string
  value: unknown
  emptyMessage?: string
  className?: string
}

const TOKEN =
  /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"\s*:?)|\b(?:true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g

function classNameFor(token: string): string {
  if (token.startsWith('"')) {
    return token.trimEnd().endsWith(":") ? styles.key : styles.string
  }
  if (token === "null") {
    return styles.null
  }
  return styles.number
}

function highlight(json: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const pattern = new RegExp(TOKEN)
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(json)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(json.slice(lastIndex, match.index))
    }

    nodes.push(
      <span key={match.index} className={classNameFor(match[0])}>
        {match[0]}
      </span>,
    )

    lastIndex = match.index + match[0].length
  }

  if (lastIndex < json.length) {
    nodes.push(json.slice(lastIndex))
  }

  return nodes
}

export function JsonPanel({
  title = "Output JSON",
  value,
  emptyMessage = "No output yet — start a run to see the response here.",
  className,
}: JsonPanelProps) {
  const [copied, setCopied] = useState(false)
  const [hidden, setHidden] = useState(true)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    return () => {
      if (timer.current !== undefined) {
        window.clearTimeout(timer.current)
      }
    }
  }, [])

  if (hidden) {
    return (
      <Button
        icon={Eye}
        className={className}
        onClick={() => setHidden(false)}
      >
        Show JSON
      </Button>
    )
  }

  if (value === null || value === undefined) {
    return (
      <Panel title={title} className={className}>
        <p className={styles.empty}>{emptyMessage}</p>
      </Panel>
    )
  }

  const json = JSON.stringify(value, null, 2)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(json)
      setCopied(true)
      timer.current = window.setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard access can be denied; the JSON stays selectable by hand.
    }
  }

  return (
    <Panel
      title={title}
      className={className}
      actions={
        <>
          <Button icon={EyeOff} onClick={() => setHidden(true)}>
            Hide
          </Button>
          <Button icon={copied ? Check : Copy} onClick={handleCopy}>
            {copied ? "Copied" : "Copy JSON"}
          </Button>
        </>
      }
    >
      <pre className={styles.code}>{highlight(json)}</pre>
    </Panel>
  )
}
export function OutputPanel({
  running,
  error,
  result,
}: {
  running: boolean
  error: string | null
  result: unknown
}) {
  let content: string
  if (error) content = `Error: ${error}`
  else if (running) content = "Running… this can take a while."
  else if (result) content = JSON.stringify(result, null, 2)
  else content = "Idle — fill the form and press Start."

  return (
    <div style={{ marginTop: 16 }}>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 6 }}>Output</p>
      <div
        style={{
          background: "var(--surface-2)",
          border: "0.5px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: 14,
          minHeight: 60,
          fontFamily: "var(--font-mono)",
          fontSize: 12.5,
          color: error ? "var(--danger-text)" : "var(--text-primary)",
          whiteSpace: "pre-wrap",
        }}
      >
        {content}
      </div>
    </div>
  )
}
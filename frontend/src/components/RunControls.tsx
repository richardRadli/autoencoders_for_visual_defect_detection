import { Button } from "./Button"

export function RunControls({
  running,
  onStart,
  onStop,
  startLabel,
}: {
  running: boolean
  onStart: () => void
  onStop: () => void
  startLabel: string
}) {
  return (
    <div style={{ display: "flex", gap: 8 }}>
      <Button variant="primary" onClick={onStart} disabled={running}>
        {running ? "Running…" : startLabel}
      </Button>
      <Button variant="danger" onClick={onStop} disabled={!running}>
        Stop
      </Button>
    </div>
  )
}
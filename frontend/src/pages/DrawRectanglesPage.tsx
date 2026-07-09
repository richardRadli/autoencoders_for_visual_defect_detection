import { Link } from "react-router-dom"

export function DrawRectanglesPage() {
  return (
    <div>
      <Link to="/" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>← Back to menu</Link>
      <h1 style={{ fontSize: 22, fontWeight: 500, marginTop: 12 }}>Draw rectangles</h1>
    </div>
  )
}
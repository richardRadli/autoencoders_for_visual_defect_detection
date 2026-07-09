import { Link } from "react-router-dom"

export function TestingPage() {
  return (
    <div>
      <Link to="/" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>← Back to menu</Link>
      <h1 style={{ fontSize: 22, fontWeight: 500, marginTop: 12 }}>Testing</h1>
    </div>
  )
}
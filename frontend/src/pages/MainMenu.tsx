import { Link } from "react-router-dom"

const linkStyle = { color: "var(--accent)", textDecoration: "none" }

export function MainMenu() {
  return (
    <div>
      <h1 style={{ fontSize: 22, fontWeight: 500, marginBottom: 16 }}>Main menu</h1>
      <nav style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 260 }}>
        <Link to="/augmentation" style={linkStyle}>Augmentation</Link>
        <Link to="/draw-rectangles" style={linkStyle}>Draw rectangles</Link>
        <Link to="/training" style={linkStyle}>Training</Link>
        <Link to="/testing" style={linkStyle}>Testing</Link>
      </nav>
    </div>
  )
}
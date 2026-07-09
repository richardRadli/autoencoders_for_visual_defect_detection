import { Header } from "./components/Header"

function App() {
  return (
    <div>
      <Header />
      <main style={{ padding: 24 }}>
        <p style={{ color: "var(--text-secondary)" }}>Theme test — click the icon top-right.</p>
      </main>
    </div>
  )
}

export default App
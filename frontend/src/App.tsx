import { Routes, Route } from "react-router-dom"
import { Header } from "./components/Header"
import { Footer } from "./components/Footer"
import { MainMenu } from "./pages/MainMenu"
import { AugmentationPage } from "./pages/AugmentationPage"
import { DrawRectanglesPage } from "./pages/DrawRectanglesPage"
import { TrainingPage } from "./pages/TrainingPage"
import { TestingPage } from "./pages/TestingPage"

function App() {
  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <Header />
      <main style={{ flex: 1, padding: 24 }}>
        <Routes>
          <Route path="/" element={<MainMenu />} />
          <Route path="/augmentation" element={<AugmentationPage />} />
          <Route path="/draw-rectangles" element={<DrawRectanglesPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/testing" element={<TestingPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App
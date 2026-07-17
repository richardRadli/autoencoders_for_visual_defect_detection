import { Navigate, Route, Routes } from "react-router-dom"
import { AugmentationPage } from "./pages/AugmentationPage"
import { Footer } from "./components/Footer/Footer"
import { Header } from "./components/Header/Header"
import { DrawRectanglesPage } from "./pages/DrawRectanglesPage"
import { MainMenu } from "./pages/MainMenu"
import { TestingPage } from "./pages/TestingPage"
import { TrainingPage } from "./pages/TrainingPage"
import styles from "./App.module.css"

function App() {
  return (
    <div className={styles.app}>
      <Header />

      <main className={styles.main}>
        <div className={styles.content}>
          <Routes>
            <Route path="/" element={<MainMenu />} />
            <Route path="/draw-rectangles" element={<DrawRectanglesPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/testing" element={<TestingPage />} />
            <Route path="/augmentation" element={<AugmentationPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default App
import { useNavigate } from "react-router-dom"
import { Button } from "./Button"

export function BackToMenu() {
  const navigate = useNavigate()
  return <Button onClick={() => navigate("/")}>← Back to menu</Button>
}